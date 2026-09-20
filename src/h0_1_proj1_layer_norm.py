"""
H0.1 sub-hypothesis: FlashHead's AudioProjModel.proj1 is a Linear(46080→512)
where input is flatten of (window=5, layers=12, dim=768) = 5*12*768 = 46080.

If we group input columns by (layer, dim) and compute per-layer weight L2 norm,
we discover which wav2vec2 layers FlashHead actually relies on.

Interpretation:
  - early-layer norms HIGH → FlashHead reads acoustic features → run_006 early-layer
    gap (cos 0.60) directly hurts Phase 1, layer-gate Bridge MLP has clear headroom
  - late-layer norms HIGH → FlashHead reads semantic features → run_006 late-layer
    cos (0.99) means TTS≈real, Phase 1 already good, Bridge MLP has nothing to fix
  - flat norms → FlashHead averages all layers → mixed expectation
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .project_paths import FLASHHEAD_ROOT
else:
    from project_paths import FLASHHEAD_ROOT

import numpy as np
import torch


CKPT_DIR = FLASHHEAD_ROOT / "models" / "SoulX-FlashHead-1_3B"


def load_audio_proj_weight():
    """Load only the AudioProjModel weights — much smaller than full DiT.
    Returns proj1.weight shape (intermediate_dim=512, input_dim=46080).
    """
    sys.path.insert(0, str(FLASHHEAD_ROOT))
    from flash_head.src.modules.flash_head_model import AudioProjModel

    # Construct AudioProjModel with FlashHead's standard config
    # From flash_head_model.py:384-391:
    #   audio_window=5, intermediate_dim=512, output_dim=1536,
    #   context_tokens=32, norm_output_audio=True
    # vae_scale=4 → seq_len_vf = 5+4-1 = 8
    audio_proj = AudioProjModel(
        seq_len=5,
        seq_len_vf=8,
        intermediate_dim=512,
        output_dim=1536,
        context_tokens=32,
        norm_output_audio=True,
    )

    # Find the AudioProjModel weights inside the FlashHead checkpoint
    import glob
    safetensor_files = sorted(glob.glob(str(CKPT_DIR / "**" / "*.safetensors"), recursive=True))
    if not safetensor_files:
        # try .pt or .bin
        bin_files = sorted(glob.glob(str(CKPT_DIR / "**" / "*.pt"), recursive=True))
        bin_files += sorted(glob.glob(str(CKPT_DIR / "**" / "*.bin"), recursive=True))
        print(f"[h0_1] Available checkpoint files: {bin_files[:5]} (showing first 5)")
        return audio_proj, None

    print(f"[h0_1] Found {len(safetensor_files)} safetensor files in {CKPT_DIR}")
    print(f"[h0_1] First: {safetensor_files[0]}")

    from safetensors.torch import load_file
    all_keys_with_audio = []
    for f in safetensor_files:
        sd = load_file(f, device="cpu")
        for k in sd.keys():
            if "audio_proj" in k.lower():
                all_keys_with_audio.append((f, k, tuple(sd[k].shape)))
    return audio_proj, all_keys_with_audio


def analyze_proj1_per_layer(weight: torch.Tensor, num_layers: int = 12, dim: int = 768, window: int = 5):
    """proj1.weight shape: (out_dim=intermediate, in_dim=window*layers*dim)
    Reshape to (out_dim, window, layers, dim) then compute L2 norm grouped by layer.
    """
    out_dim, in_dim = weight.shape
    expected_in = window * num_layers * dim
    assert in_dim == expected_in, f"in_dim {in_dim} != {window}*{num_layers}*{dim}={expected_in}"

    w = weight.view(out_dim, window, num_layers, dim)  # (out_dim, 5, 12, 768)

    # per-layer norm: ||w[:, :, l, :]||_F  (frobenius, over out_dim/window/dim)
    per_layer_norm = torch.zeros(num_layers)
    for l in range(num_layers):
        per_layer_norm[l] = w[:, :, l, :].pow(2).sum().sqrt()
    per_layer_norm = per_layer_norm / per_layer_norm.sum()

    # per-window-position norm
    per_window_norm = torch.zeros(window)
    for i in range(window):
        per_window_norm[i] = w[:, i, :, :].pow(2).sum().sqrt()
    per_window_norm = per_window_norm / per_window_norm.sum()

    return {
        "per_layer_share": [float(x) for x in per_layer_norm],
        "per_window_share": [float(x) for x in per_window_norm],
    }


def main():
    print("=== H0.1 · FlashHead AudioProjModel proj1 per-layer weight analysis ===")
    audio_proj, keys = load_audio_proj_weight()

    if keys is None:
        print("[h0_1] No safetensors found — trying random-init analysis as fallback")
        # at random init, expect roughly uniform per-layer
        proj1_w = audio_proj.proj1.weight.detach()
        print(f"[h0_1] proj1 weight shape (RANDOM INIT): {tuple(proj1_w.shape)}")
        result = analyze_proj1_per_layer(proj1_w)
    else:
        print(f"[h0_1] Found {len(keys)} audio_proj keys in checkpoints:")
        for f, k, s in keys[:10]:
            print(f"  {Path(f).name} :: {k} {s}")

        # Find proj1 weight specifically
        from safetensors.torch import load_file
        proj1_w = None
        for f, k, s in keys:
            if "proj1" in k and "weight" in k and "norm" not in k:
                sd = load_file(f, device="cpu")
                proj1_w = sd[k].float()
                print(f"\n[h0_1] Loading {k} from {Path(f).name}, shape {tuple(proj1_w.shape)}")
                break

        if proj1_w is None:
            print("[h0_1] No proj1 weight found in checkpoint, using random init")
            proj1_w = audio_proj.proj1.weight.detach()
        result = analyze_proj1_per_layer(proj1_w)

    print("\n=== Per-wav2vec2-layer weight share (sums to 1.0) ===")
    print("layer | share | bar")
    print("------|-------|" + "-" * 40)
    max_share = max(result["per_layer_share"])
    for l, share in enumerate(result["per_layer_share"]):
        bar = "█" * int(share / max_share * 40)
        marker = " ←" if share > 0.10 else ""
        print(f"  {l+1:3d} | {share:.4f} | {bar}{marker}")

    print("\n=== Per-window-position weight share (5 frames) ===")
    print("  window 0 (oldest) — window 4 (newest)")
    for i, share in enumerate(result["per_window_share"]):
        bar = "█" * int(share * 100)
        print(f"  pos {i}: {share:.4f} | {bar}")

    # save to data/
    out_path = Path("data/h0_1_proj1_layer_norm.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# H0.1 · FlashHead AudioProjModel proj1 per-layer weight analysis\n\n")
        f.write("Per-wav2vec2-layer weight L2-norm share:\n")
        for l, share in enumerate(result["per_layer_share"]):
            f.write(f"  layer {l+1:3d}: {share:.4f}\n")
        f.write(f"\nargmax = layer {1 + np.argmax(result['per_layer_share'])}\n")
        f.write(f"max share = {max(result['per_layer_share']):.4f}\n")
        f.write(f"min share = {min(result['per_layer_share']):.4f}\n")
        f.write(f"max/min ratio = {max(result['per_layer_share'])/min(result['per_layer_share']):.2f}\n")
    print(f"\n[h0_1] Saved → {out_path}")


if __name__ == "__main__":
    main()
