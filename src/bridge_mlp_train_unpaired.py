"""
Unpaired domain-matching training for Bridge MLP.

Strategy: BridgeMLP(syn_features) should match the *distribution* of real features.
We don't have paired data, so we match per-layer mean and std (1st + 2nd moments).
Add residual regularizer so output doesn't drift too far from input.

Day 1 of interview-deliverable plan: produce first real H0 vs B1 cos numbers.

Usage:
    python src/bridge_mlp_train_unpaired.py --device cuda --epochs 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from bridge_mlp import BridgeMLPLight  # noqa: E402


DATA = Path("data")
TTS_FEAT_DIR = DATA / "tts_batch_features"
OUT_DIR = Path("experiments/H0-bridge-mlp-baseline/results")


class FeaturesDataset(Dataset):
    """Loads pre-extracted (T, 12, 768) features. Random crop to clip_frames."""

    def __init__(self, files: list[Path], clip_frames: int = 33):
        self.files = files
        self.clip_frames = clip_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        feats = np.load(self.files[i])  # (T, 12, 768)
        T = feats.shape[0]
        if T > self.clip_frames:
            start = np.random.randint(0, T - self.clip_frames + 1)
            feats = feats[start : start + self.clip_frames]
        elif T < self.clip_frames:
            pad = self.clip_frames - T
            feats = np.concatenate([feats, np.zeros((pad, 12, 768), dtype=feats.dtype)], 0)
        return torch.from_numpy(feats).float()


def distribution_match_loss(pred, target_pool):
    """Match per-layer 1st + 2nd moments between pred and target_pool.

    pred:        (B, T, L, D) — Bridge MLP output on synthetic features
    target_pool: (B', T', L, D) — real features (sampled from real pool)
    Returns scalar loss.
    """
    # flatten to (B*T, L, D) for moment computation
    p = pred.reshape(-1, pred.shape[-2], pred.shape[-1])         # (N, L, D)
    t = target_pool.reshape(-1, target_pool.shape[-2], target_pool.shape[-1])  # (M, L, D)

    p_mean = p.mean(dim=0)   # (L, D)
    t_mean = t.mean(dim=0)   # (L, D)
    p_std = p.std(dim=0) + 1e-6
    t_std = t.std(dim=0) + 1e-6

    loss_mean = F.mse_loss(p_mean, t_mean)
    loss_std = F.mse_loss(p_std, t_std)
    return loss_mean + loss_std, {"l_mean": float(loss_mean), "l_std": float(loss_std)}


def cos_sim_layerwise(features_a, features_b):
    """Per-layer mean cosine similarity. Inputs (B, T, L, D)."""
    a = features_a.reshape(-1, features_a.shape[-2], features_a.shape[-1]).mean(0)  # (L, D)
    b = features_b.reshape(-1, features_b.shape[-2], features_b.shape[-1]).mean(0)
    cos = F.cosine_similarity(a, b, dim=-1)  # (L,)
    return cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_residual", type=float, default=0.5)
    parser.add_argument("--clip_frames", type=int, default=33)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------- Load data -------
    syn_files = sorted(TTS_FEAT_DIR.glob("*.npy"))
    real_files = sorted(DATA.glob("spike_*_features.npy"))
    real_files = [f for f in real_files if "tts" not in f.name]
    print(f"[train] {len(syn_files)} synthetic clips, {len(real_files)} real clips")
    assert len(syn_files) > 0 and len(real_files) > 0

    syn_ds = FeaturesDataset(syn_files, clip_frames=args.clip_frames)
    syn_loader = DataLoader(syn_ds, batch_size=args.bs, shuffle=True, num_workers=0)

    # Real features: load all into memory once (only ~150MB total)
    real_pool = []
    for f in real_files:
        feats = np.load(f)  # (T, 12, 768)
        # if T very long (e.g. podcast 1648), random-crop multiple chunks per epoch
        real_pool.append(torch.from_numpy(feats).float())
    print(f"[train] real_pool clips: {[r.shape for r in real_pool]}")

    def sample_real_chunk(n_chunks=8):
        """Sample n random clip_frames-length chunks from real pool."""
        out = []
        for _ in range(n_chunks):
            r = real_pool[np.random.randint(len(real_pool))]
            T = r.shape[0]
            if T <= args.clip_frames:
                out.append(F.pad(r, (0, 0, 0, 0, 0, args.clip_frames - T)))
            else:
                start = np.random.randint(0, T - args.clip_frames + 1)
                out.append(r[start : start + args.clip_frames])
        return torch.stack(out)  # (n_chunks, clip_frames, 12, 768)

    # ------- Model -------
    model = BridgeMLPLight(num_layers=12, dim=768, residual=True, layer_gate=True).to(args.device)
    model.init_as_identity()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] BridgeMLPLight params: {n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ------- Initial metrics (B1 baseline = no Bridge MLP) -------
    print("\n=== Initial state (= B1 baseline, no domain adaptation) ===")
    model.eval()
    with torch.no_grad():
        syn_batch_init = next(iter(syn_loader)).to(args.device)
        real_batch_init = sample_real_chunk(8).to(args.device)
        cos_init = cos_sim_layerwise(syn_batch_init, real_batch_init)
        for l, c in enumerate(cos_init):
            print(f"  layer {l+1:2d}: cos(syn, real) = {float(c):.4f}")
    model.train()

    # ------- Training -------
    log = []
    step = 0
    for epoch in range(args.epochs):
        for syn_batch in syn_loader:
            syn_batch = syn_batch.to(args.device)         # (B, T, 12, 768)
            real_batch = sample_real_chunk(8).to(args.device)

            pred = model(syn_batch)                        # (B, T, 12, 768)
            loss_dist, dlogs = distribution_match_loss(pred, real_batch)
            loss_residual = F.mse_loss(pred, syn_batch)
            loss = loss_dist + args.lambda_residual * loss_residual

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            log_entry = {
                "step": step,
                "epoch": epoch,
                "loss_total": float(loss),
                "loss_residual": float(loss_residual),
                **dlogs,
                "gate_mean": float(torch.sigmoid(model.gate).mean()),
                "gate_layer1": float(torch.sigmoid(model.gate[0])),
                "gate_layer12": float(torch.sigmoid(model.gate[-1])),
            }
            log.append(log_entry)
            if step % 5 == 0:
                print(f"  step {step:3d} | loss {float(loss):.4f} "
                      f"| dist_mean {dlogs['l_mean']:.4f} dist_std {dlogs['l_std']:.4f} "
                      f"| residual {float(loss_residual):.4f} "
                      f"| gate avg {torch.sigmoid(model.gate).mean():.3f} "
                      f"L1 {torch.sigmoid(model.gate[0]):.3f} L12 {torch.sigmoid(model.gate[-1]):.3f}")
            step += 1

    # ------- Final metrics (B2 = with Bridge MLP) -------
    print("\n=== Final state (= B2 with layer-gated Bridge MLP) ===")
    model.eval()
    with torch.no_grad():
        # average over multiple batches for stable measurement
        cos_final_layers = torch.zeros(12)
        n_eval = 8
        for _ in range(n_eval):
            syn_batch = next(iter(syn_loader)).to(args.device)
            real_batch = sample_real_chunk(8).to(args.device)
            pred = model(syn_batch)
            cos_final_layers += cos_sim_layerwise(pred, real_batch).cpu()
        cos_final_layers /= n_eval

        cos_b1_layers = torch.zeros(12)
        for _ in range(n_eval):
            syn_batch = next(iter(syn_loader)).to(args.device)
            real_batch = sample_real_chunk(8).to(args.device)
            cos_b1_layers += cos_sim_layerwise(syn_batch, real_batch).cpu()
        cos_b1_layers /= n_eval

    print(f"\n=== B1 vs B2 per-layer cosine (averaged over {n_eval} batches) ===")
    print(f"layer | B1 (raw)  | B2 (Bridge) | delta   | gate")
    print(f"------|-----------|-------------|---------|------")
    for l in range(12):
        gate = float(torch.sigmoid(model.gate[l]))
        delta = float(cos_final_layers[l] - cos_b1_layers[l])
        marker = " ✓" if delta > 0.005 else ("  ✗" if delta < -0.005 else "")
        print(f"  {l+1:3d} | {float(cos_b1_layers[l]):.4f}    | {float(cos_final_layers[l]):.4f}      "
              f"| {delta:+.4f} | {gate:.3f}{marker}")

    # save
    torch.save(model.state_dict(), OUT_DIR / "bridge_mlp_unpaired.pt")
    with open(OUT_DIR / "training_log.json", "w") as f:
        json.dump({
            "config": vars(args),
            "n_params": n_params,
            "log": log,
            "cos_b1_per_layer": [float(x) for x in cos_b1_layers],
            "cos_b2_per_layer": [float(x) for x in cos_final_layers],
            "gate_per_layer": [float(torch.sigmoid(model.gate[l])) for l in range(12)],
        }, f, indent=2)
    print(f"\n[train] Saved → {OUT_DIR}/bridge_mlp_unpaired.pt + training_log.json")


if __name__ == "__main__":
    main()
