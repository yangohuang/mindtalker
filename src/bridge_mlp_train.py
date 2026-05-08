"""
Bridge MLP training loop skeleton for H0.

Status: scaffolding (no real training yet). Decisions documented inline.

Training data design (see protocol.md → 4-tier baseline matrix):
  - Domain shift target: align distribution of MiniCPM-o-emitted-audio's wav2vec2 features
    to FlashHead-training-distribution wav2vec2 features.
  - Source pairs: (a) MiniCPM-o "speaks" sentence S → audio_A → wav2vec2(audio_A) → src_feat
                  (b) Real human speaks S → audio_B → wav2vec2(audio_B) → tgt_feat
  - Bridge MLP learns: src_feat → tgt_feat
  - 5000-10000 paired sentences from AISHELL-3 (Chinese) — see val_set_plan.md

Loss design:
  primary  = (1 - cos_sim(pred, target)).mean()                    (per-frame, per-layer)
  reg      = lambda * MSE(pred, target) / target.var(dim=-1)       (variance-normalized)
  identity = mu * MSE(pred, src) when warmup_step < 100             (don't drift far early)

Total: loss = primary + 0.5 * reg + 0.3 * identity
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from bridge_mlp import BridgeMLPLight  # noqa: E402


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/paired_features")
    output_dir: Path = Path("experiments/H0-bridge-mlp-baseline/results")
    batch_size: int = 8
    num_epochs: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    device: str = "cuda"
    seed: int = 42
    log_every: int = 20
    save_every: int = 500
    # weights
    w_cos: float = 1.0
    w_mse_norm: float = 0.5
    w_identity: float = 0.3


class PairedFeaturesDataset(Dataset):
    """Loads pre-extracted (src_feat, tgt_feat) numpy pairs.

    Expected directory layout:
        data/paired_features/
            train/
                pair_000000_src.npy   # (T, 12, 768)  — MiniCPM-o-derived
                pair_000000_tgt.npy   # (T, 12, 768)  — Real-audio-derived
                ...
            val/
                ...
    """

    def __init__(self, dir_: Path, split: str = "train", clip_frames: int = 33):
        self.dir = dir_ / split
        self.clip_frames = clip_frames
        self.indices = sorted(
            int(p.stem.split("_")[1])
            for p in self.dir.glob("pair_*_src.npy")
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        src = np.load(self.dir / f"pair_{idx:06d}_src.npy")
        tgt = np.load(self.dir / f"pair_{idx:06d}_tgt.npy")
        # crop to clip_frames (random start)
        T = min(src.shape[0], tgt.shape[0])
        if T > self.clip_frames:
            start = np.random.randint(0, T - self.clip_frames + 1)
            src = src[start : start + self.clip_frames]
            tgt = tgt[start : start + self.clip_frames]
        elif T < self.clip_frames:
            # pad
            pad = self.clip_frames - T
            src = np.concatenate([src, np.zeros((pad, 12, 768), dtype=src.dtype)], 0)
            tgt = np.concatenate([tgt, np.zeros((pad, 12, 768), dtype=tgt.dtype)], 0)
        return torch.from_numpy(src).float(), torch.from_numpy(tgt).float()


def compute_loss(pred, target, src, step: int, cfg: TrainConfig):
    cos = F.cosine_similarity(pred, target, dim=-1)         # (B, T, L)
    loss_cos = (1.0 - cos).mean()

    # variance-normalized MSE
    var = target.var(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, T, L, 1)
    loss_mse = ((pred - target) ** 2 / var).mean()

    # identity warmup — keep close to source early in training
    if step < cfg.warmup_steps:
        loss_identity = F.mse_loss(pred, src)
    else:
        loss_identity = torch.tensor(0.0, device=pred.device)

    total = (
        cfg.w_cos * loss_cos
        + cfg.w_mse_norm * loss_mse
        + cfg.w_identity * loss_identity
    )
    return total, {
        "loss/total": float(total.item()),
        "loss/cos": float(loss_cos.item()),
        "loss/mse_norm": float(loss_mse.item()),
        "loss/identity": float(loss_identity.item()) if step < cfg.warmup_steps else 0.0,
    }


def train_one_step(model, batch, optim, step, cfg):
    src, tgt = batch
    src = src.to(cfg.device)
    tgt = tgt.to(cfg.device)
    pred = model(src)
    loss, logs = compute_loss(pred, tgt, src, step, cfg)
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optim.step()
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/paired_features"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/H0-bridge-mlp-baseline/results"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test with random tensors, skip dataset")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir, output_dir=args.output_dir,
        batch_size=args.bs, num_epochs=args.epochs, lr=args.lr, device=args.device,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = BridgeMLPLight().to(cfg.device)
    model.init_as_identity()  # start near identity to avoid distribution drift
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if args.dry_run:
        print("[dry-run] Using random tensors as data.")
        for step in range(5):
            src = torch.randn(cfg.batch_size, 33, 12, 768, device=cfg.device)
            tgt = src + 0.1 * torch.randn_like(src)  # small distribution shift
            logs = train_one_step(model, (src, tgt), optim, step, cfg)
            print(f"step {step}:", {k: f"{v:.4f}" for k, v in logs.items()})
        return

    train_ds = PairedFeaturesDataset(cfg.data_dir, split="train")
    val_ds = PairedFeaturesDataset(cfg.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    print(f"[train] {len(train_ds)} train pairs, {len(val_ds)} val pairs")
    step = 0
    for epoch in range(cfg.num_epochs):
        for batch in train_loader:
            logs = train_one_step(model, batch, optim, step, cfg)
            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step}:", {k: f"{v:.4f}" for k, v in logs.items()})
            if step % cfg.save_every == 0 and step > 0:
                ckpt = cfg.output_dir / f"bridge_mlp_step{step}.pt"
                torch.save(model.state_dict(), ckpt)
                print(f"[train] saved → {ckpt}")
            step += 1

    final = cfg.output_dir / "bridge_mlp_final.pt"
    torch.save(model.state_dict(), final)
    print(f"[train] DONE. Final → {final}")


if __name__ == "__main__":
    main()
