"""
W2VHead training (H1 spike).

Strategy: pseudo-paired — for each Chinese sentence in our prompt set:
  src: text → MiniMind-O Thinker bridge_layer 3 hidden state
  tgt: same text → Qwen3-TTS audio → wav2vec2 12-layer features

The "real" version of H1 would use real human audio for tgt; we approximate
with TTS to bootstrap. (TTS is what we have today; AISHELL-3 is 12GB to download.)

Loss = per-layer mean+std distribution match + token-level cosine similarity
       (sentence-aligned, not frame-aligned — temporal alignment is fuzzy)

Run: ~50 epoch / 5 min / single 4090.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from .project_paths import PROJECT_ROOT, MINIMIND_REPO
else:
    from project_paths import PROJECT_ROOT, MINIMIND_REPO

import numpy as np
import torch
import torch.nn.functional as F


RESEARCH = PROJECT_ROOT
sys.path.insert(0, str(RESEARCH / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--save_dir", type=Path,
                         default=RESEARCH / "experiments/H1-w2v-head/results")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ---- Load MiniMind-O once ----
    sys.path.insert(0, str(MINIMIND_REPO))
    import types, importlib.machinery
    for mod_name in ("onnxruntime", "funasr"):
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
            m.AutoModel = lambda *a, **k: None
            sys.modules[mod_name] = m

    from model.model_omni import MiniMindOmni, OmniConfig
    from transformers import AutoTokenizer

    WEIGHT = MINIMIND_REPO / "out"
    print(f"[h1train] Loading MiniMind-O from {WEIGHT}")
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHT), trust_remote_code=True)
    cfg = OmniConfig()
    minimind = MiniMindOmni(config=cfg,
                             audio_encoder_path="/nonexistent",
                             vision_model_path="/nonexistent")
    sd = torch.load(str(WEIGHT / "pytorch_model.bin"), map_location="cpu", weights_only=False)
    minimind.load_state_dict(sd, strict=False)
    minimind.audio_encoder = None
    minimind.vision_encoder = None
    minimind = minimind.half().to(device).eval()
    bridge_layer_idx = minimind.config.bridge_layer
    print(f"[h1train] MiniMind-O loaded, bridge_layer={bridge_layer_idx}")

    @torch.no_grad()
    def get_bridge(prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = torch.tensor(tokenizer(txt).data["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        captured = {}
        def hook(_, __, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["bridge"] = h.detach().float()
        handle = minimind.thinker.layers[bridge_layer_idx].register_forward_hook(hook)
        _ = minimind(input_ids=ids)
        handle.remove()
        return captured["bridge"]  # (1, T_text, 768)

    # ---- Build pseudo-pair dataset ----
    # Use sentences from batch_tts_features.sh (already have features cached)
    # Each sentence has: text → bridge   AND   tts_audio → wav2vec2 features
    sentences = [
        "今天天气不错，适合出门散步，我们去公园走一走吧。",
        "最近的人工智能技术发展非常迅速，特别是大语言模型领域。",
        "这家咖啡店的拿铁口感很好，价格也不贵，推荐给大家。",
        "我们公司昨天开了一个会议，主要讨论了下一季度的产品规划。",
        "深圳的春天总是来得比较早，三月初就能感受到温暖的气息。",
    ]
    # use first 5 sentences × 5 voices = 25 pairs (we have 25 sentences × 5 voices = 125 features)

    feat_dir = RESEARCH / "data/tts_batch_features"
    pairs = []  # list of (bridge_states, target_features)
    print(f"[h1train] Building pseudo-pairs...")
    for sent_idx, prompt in enumerate(sentences):
        bridge = get_bridge(prompt)  # (1, T_text, 768)
        # pair with all 5 voices' features for this sentence
        for voice in ["vivian", "serena", "dylan", "eric", "ryan"]:
            feat_path = feat_dir / f"{voice}_clip_{sent_idx:03d}.npy"
            if feat_path.exists():
                target = np.load(feat_path)  # (T_audio, 12, 768)
                pairs.append((bridge.squeeze(0).cpu(), torch.from_numpy(target).float()))
    print(f"[h1train] {len(pairs)} pseudo-pairs built")

    # free up GPU MiniMind-O memory
    del minimind
    torch.cuda.empty_cache()

    # ---- Init W2VHead ----
    from h1_w2v_head import W2VHead
    head = W2VHead(hidden_size=768, num_layers=12).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[h1train] W2VHead params: {n_params/1e6:.2f}M")
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    # ---- Initial eval (random init) ----
    head.eval()
    initial_cos = []
    with torch.no_grad():
        for bridge, target in pairs[:8]:
            bridge = bridge.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            T_audio = target.shape[1]
            pred = head(bridge, target_audio_frames=T_audio)
            # per-layer mean cosine
            for l in range(12):
                a = pred[0, :, l, :].mean(0)
                b = target[0, :, l, :].mean(0)
                if l == 0 and len(initial_cos) == 0:
                    initial_cos = [F.cosine_similarity(a, b, dim=0).item() for l in range(12)]
                    break
    head.train()

    # ---- Training ----
    log = []
    step = 0
    print(f"[h1train] Training {args.epochs} epoch × {len(pairs)} pairs...")
    for epoch in range(args.epochs):
        np.random.shuffle(pairs)
        epoch_loss = []
        for bridge_cpu, target_cpu in pairs:
            bridge = bridge_cpu.unsqueeze(0).to(device)        # (1, T_text, 768)
            target = target_cpu.unsqueeze(0).to(device)        # (1, T_audio, 12, 768)
            T_audio = target.shape[1]

            pred = head(bridge, target_audio_frames=T_audio)   # (1, T_audio, 12, 768)

            # per-layer cosine loss
            loss_cos = 0.0
            for l in range(12):
                p = pred[0, :, l, :]
                t = target[0, :, l, :]
                # per-frame cos sim (averaged)
                sim = F.cosine_similarity(p, t, dim=-1).mean()
                loss_cos = loss_cos + (1.0 - sim)
            loss_cos = loss_cos / 12

            # variance-normalized MSE
            var = target.var(dim=1, keepdim=True).clamp(min=1e-6)  # (1, 1, 12, 768)
            loss_mse = ((pred - target) ** 2 / var).mean()

            loss = loss_cos + 0.5 * loss_mse

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()

            epoch_loss.append(float(loss))
            step += 1

        if (epoch + 1) % 5 == 0:
            head.eval()
            with torch.no_grad():
                cos_per_layer = torch.zeros(12)
                n = 0
                for bridge_cpu, target_cpu in pairs[:8]:
                    bridge = bridge_cpu.unsqueeze(0).to(device)
                    target = target_cpu.unsqueeze(0).to(device)
                    T_audio = target.shape[1]
                    pred = head(bridge, target_audio_frames=T_audio)
                    for l in range(12):
                        a = pred[0, :, l, :].mean(0); a = a/(a.norm()+1e-8)
                        b = target[0, :, l, :].mean(0); b = b/(b.norm()+1e-8)
                        cos_per_layer[l] += (a * b).sum().item()
                    n += 1
                cos_per_layer /= n
            head.train()

            print(f"  epoch {epoch+1:3d} | loss {np.mean(epoch_loss):.4f} | "
                  f"L1 {cos_per_layer[0]:.3f} L6 {cos_per_layer[5]:.3f} L11 {cos_per_layer[10]:.3f} L12 {cos_per_layer[11]:.3f}")

            log.append({"epoch": epoch + 1, "loss": float(np.mean(epoch_loss)),
                        "cos_per_layer": [float(x) for x in cos_per_layer]})

    # save
    torch.save(head.state_dict(), args.save_dir / "w2v_head_trained.pt")
    with open(args.save_dir / "training_log.json", "w") as f:
        json.dump({"epochs": args.epochs, "lr": args.lr, "bs": args.bs,
                    "n_pairs": len(pairs), "n_params": n_params, "log": log}, f, indent=2)
    print(f"\n[h1train] DONE. Saved → {args.save_dir}")


if __name__ == "__main__":
    main()
