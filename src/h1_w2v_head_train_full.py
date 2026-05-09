"""
Stage A/B/C · W2VHead 大数据训练 (full version)

输入：data/paired/*.npz （由 build_paired_dataset.py 生成）
输出：experiments/H1-w2v-head/results/w2v_head_full_{epoch}.pt

Pipeline:
  1. DataLoader：每条 sample = (text, real_w2v_features); collate 批处理
  2. MiniMind-O Thinker (frozen) on GPU → bridge_states for the batch's text
  3. W2VHead → predicted features
  4. Loss = per-frame cosine + var-norm MSE (vs real features)
  5. AdamW + ReduceLROnPlateau + checkpoint each epoch

预期规模：
  - Stage A spike: 100-200 pairs / 50 epoch / 30 min
  - Stage B medium: 1000 pairs / 50 epoch / 2-3h
  - Stage C full: 5000+ pairs / 100 epoch / 10-20h

用法：
    PYTHONNOUSERSITE=1 /path/to/miniforge3/envs/flashhead/bin/python src/h1_w2v_head_train_full.py \\
        --epochs 50 --bs 16 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/path/to/yg/code/mindtalker/research")
MINIMIND_REPO = Path("/path/to/yg/code/github/minimind-o")
MINIMIND_WEIGHT = MINIMIND_REPO / "out"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MINIMIND_REPO))


def load_minimind(device):
    """Load MiniMind-O frozen for bridge_states extraction."""
    import importlib.machinery
    for m in ("onnxruntime", "funasr"):
        if m not in sys.modules:
            mod = types.ModuleType(m)
            mod.__spec__ = importlib.machinery.ModuleSpec(m, None)
            mod.AutoModel = lambda *a, **k: None
            sys.modules[m] = mod
    from model.model_omni import MiniMindOmni, OmniConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MINIMIND_WEIGHT), trust_remote_code=True)
    cfg = OmniConfig()
    model = MiniMindOmni(config=cfg, audio_encoder_path="/none", vision_model_path="/none")
    sd = torch.load(str(MINIMIND_WEIGHT / "pytorch_model.bin"), map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.audio_encoder = None
    model.vision_encoder = None
    model = model.half().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer, model.config.bridge_layer


class PairedDataset(Dataset):
    """Loads (text, real_w2v_features) from npz files."""

    def __init__(self, index_path: Path):
        with open(index_path) as f:
            self.index = json.load(f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e = self.index[i]
        d = np.load(ROOT / e["path"], allow_pickle=False)
        return {
            "text": str(d["text"]),
            "features": torch.from_numpy(d["features"]).float(),  # (T_audio, 12, 768)
            "id": e["id"],
        }


def collate_batch(batch):
    """Pad features to max T in batch."""
    texts = [b["text"] for b in batch]
    feats = [b["features"] for b in batch]
    Tmax = max(f.shape[0] for f in feats)
    padded = torch.zeros(len(batch), Tmax, 12, 768)
    masks = torch.zeros(len(batch), Tmax, dtype=torch.bool)
    for i, f in enumerate(feats):
        padded[i, :f.shape[0]] = f
        masks[i, :f.shape[0]] = True
    return {"texts": texts, "features": padded, "mask": masks}


@torch.no_grad()
def get_bridge_batch(model, tokenizer, texts, bridge_layer_idx, device):
    """Run MiniMind-O on a batch of texts, return bridge_states (B, T_text_max, 768) padded."""
    bridges = []
    Tmax = 0
    captured = []

    def hook(_, __, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach().float())

    handle = model.thinker.layers[bridge_layer_idx].register_forward_hook(hook)
    for text in texts:
        captured.clear()
        msgs = [{"role": "user", "content": text}]
        templated = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = torch.tensor(tokenizer(templated).data["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        _ = model(input_ids=ids)
        b = captured[0].squeeze(0)  # (T_text, 768)
        bridges.append(b)
        Tmax = max(Tmax, b.shape[0])
    handle.remove()

    padded = torch.zeros(len(texts), Tmax, 768, device=device)
    for i, b in enumerate(bridges):
        padded[i, :b.shape[0]] = b
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--save_dir", type=Path, default=ROOT / "experiments/H1-w2v-head/results")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ---- Load datasets ----
    index_path = ROOT / "data/paired_index.json"
    if not index_path.exists():
        print(f"[train] paired_index.json missing — run build_paired_dataset.py first")
        return
    ds = PairedDataset(index_path)
    print(f"[train] dataset size: {len(ds)} pairs")
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2,
                        collate_fn=collate_batch, drop_last=True)

    # ---- Load MiniMind-O frozen ----
    print(f"[train] Loading MiniMind-O...")
    minimind, tokenizer, bridge_layer_idx = load_minimind(device)
    print(f"[train] MiniMind-O loaded, bridge_layer={bridge_layer_idx}")

    # ---- Init W2VHead ----
    from h1_w2v_head import W2VHead
    head = W2VHead(hidden_size=768, num_layers=12).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[train] W2VHead params: {n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5,
                                                        patience=3, min_lr=1e-5)

    # ---- Train ----
    log = []
    for epoch in range(args.epochs):
        head.train()
        epoch_loss = []
        for step, batch in enumerate(loader):
            texts = batch["texts"]
            target = batch["features"].to(device)        # (B, T_audio, 12, 768)
            mask = batch["mask"].to(device)               # (B, T_audio)

            with torch.no_grad():
                bridges = get_bridge_batch(minimind, tokenizer, texts, bridge_layer_idx, device)
            T_audio = target.shape[1]
            pred = head(bridges, target_audio_frames=T_audio)   # (B, T_audio, 12, 768)

            # masked per-layer cosine loss
            loss_cos = 0.0
            for l in range(12):
                p = pred[:, :, l, :]
                t = target[:, :, l, :]
                sim = F.cosine_similarity(p, t, dim=-1)         # (B, T)
                sim = (sim * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                loss_cos = loss_cos + (1.0 - sim).mean()
            loss_cos = loss_cos / 12

            # masked var-norm MSE
            var = target.var(dim=1, keepdim=True).clamp(min=1e-6)
            mse = ((pred - target) ** 2 / var).mean(dim=-1)     # (B, T, L)
            mse = mse.mean(dim=-1)                              # (B, T)
            mse = (mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss_mse = mse.mean()

            loss = loss_cos + 0.5 * loss_mse

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()

            epoch_loss.append(float(loss))
            if step % 50 == 0:
                print(f"  ep{epoch+1:3d} step{step:4d} | loss {float(loss):.4f}")

        avg = float(np.mean(epoch_loss))
        sched.step(avg)
        log.append({"epoch": epoch + 1, "avg_loss": avg, "lr": optim.param_groups[0]["lr"]})
        print(f"[train] epoch {epoch+1:3d} | avg_loss {avg:.4f} | lr {optim.param_groups[0]['lr']:.5f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt = args.save_dir / f"w2v_head_full_ep{epoch+1:03d}.pt"
            torch.save(head.state_dict(), ckpt)
            with open(args.save_dir / "training_log_full.json", "w") as f:
                json.dump({"config": vars(args), "n_params": n_params, "log": log}, f, indent=2)
            print(f"  [save] {ckpt}")

    print(f"[train] DONE. Final ckpt + log in {args.save_dir}")


if __name__ == "__main__":
    main()
