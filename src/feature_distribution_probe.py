"""
Quantitative probe: how different are wav2vec2 features across audio sources?

H0 prerequisite question: if MiniCPM-o-derived audio's wav2vec2 features ≈ real-audio
features, Bridge MLP has nothing to learn. We don't have MiniCPM-o output yet (GPU
blocked), but we can probe whether wav2vec2 features are SENSITIVE to audio source at
all by comparing 5 different real audios:

  - podcast_sichuan_16k  (Chinese, conversation)
  - cantonese_16k        (Cantonese)
  - 2_scott_0_103_28s    (English, Scott speaker)
  - zero_shot_prompt     (Wan2.2 prompt, short)
  - talk                 (Wan2.2 example, short)

If all 5 cluster tightly in PCA space → wav2vec2 features are quite source-agnostic →
Bridge MLP space is small (conservative for H0).
If they spread → wav2vec2 features carry source-specific info → Bridge MLP space exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


DATA = Path("data")

CLIPS = [
    "spike_001",                       # podcast_sichuan, was renamed
    "spike_cantonese_16k",
    "spike_2_scott_0_103_103_28s",
    "spike_zero_shot_prompt",
    "spike_talk",
]


def load_features(stem: str) -> np.ndarray:
    """Load (T, 12, 768) features. Mean-pool over T → (12, 768) per clip signature."""
    path = DATA / f"{stem}_features.npy"
    if not path.exists():
        # try alt name
        alt = DATA / f"spike_{stem.replace('spike_', '')}_features.npy"
        if alt.exists():
            path = alt
    if not path.exists():
        return None
    feats = np.load(path)
    return feats


def per_clip_stats(name: str, feats: np.ndarray) -> dict:
    """Quick stats: mean / std / per-layer norm."""
    T, L, D = feats.shape
    return {
        "name": name,
        "T_frames": T,
        "duration_s": T / 25.0,
        "mean": float(feats.mean()),
        "std": float(feats.std()),
        "layer_norms": [float(np.linalg.norm(feats[:, l, :], axis=-1).mean()) for l in range(L)],
    }


def pca_across_clips(clip_features: dict[str, np.ndarray], n_components: int = 8):
    """Per-frame features pooled as (clip_id, frame_id) → PCA.
    Returns (n_total_frames, n_components) and clip_id labels.
    """
    all_frames = []
    labels = []
    for cid, (name, feats) in enumerate(clip_features.items()):
        # use last layer (12) as representative — closest to FlashHead's effective input
        last_layer = feats[:, -1, :]  # (T, 768)
        all_frames.append(last_layer)
        labels.extend([cid] * last_layer.shape[0])
    X = np.concatenate(all_frames, 0)
    labels = np.array(labels)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, labels, pca


def cross_clip_distances(clip_features: dict[str, np.ndarray]) -> np.ndarray:
    """Mean cosine distance between layer-12 mean features across clips."""
    names = list(clip_features.keys())
    n = len(names)
    means = np.stack(
        [clip_features[name][:, -1, :].mean(0) for name in names]
    )  # (n, 768)
    means_norm = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
    cos = means_norm @ means_norm.T  # (n, n)
    return names, cos


def main():
    # Re-glob all real spike features
    feats_files = sorted(DATA.glob("spike_*_features.npy"))
    if not feats_files:
        print("[probe] No spike features found. Run phase1_pipeline.py first.")
        return

    clip_features = {}
    print(f"[probe] Found {len(feats_files)} feature files:")
    for f in feats_files:
        feats = np.load(f)
        name = f.stem.replace("spike_", "").replace("_features", "")
        clip_features[name] = feats
        stats = per_clip_stats(name, feats)
        print(f"  {name}: T={stats['T_frames']} ({stats['duration_s']:.1f}s) "
              f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
              f"layer12_norm={stats['layer_norms'][-1]:.3f}")

    print()
    print("=== Cross-clip cosine similarity (layer-12 mean) ===")
    names, cos = cross_clip_distances(clip_features)
    print("clips:", names)
    np.set_printoptions(precision=3, suppress=True)
    print(cos)
    print(f"\noff-diagonal mean cos = {(cos.sum() - len(names)) / (len(names) * (len(names) - 1)):.4f}")
    print(f"min off-diagonal     = {(cos - 2 * np.eye(len(names))).max():.4f}")

    print("\n=== PCA across all frames (per-frame, layer-12) ===")
    X_pca, labels, pca = pca_across_clips(clip_features, n_components=8)
    print(f"explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

    # per-clip PCA centroid
    print("\n=== Per-clip PCA-8 centroid ===")
    for cid, name in enumerate(names):
        centroid = X_pca[labels == cid].mean(0)
        print(f"  {name}: {centroid[:4]}")

    # save report
    out = DATA / "feature_distribution_report.txt"
    with open(out, "w") as f:
        f.write(f"# Feature Distribution Probe — {len(names)} clips\n\n")
        f.write(f"clips: {names}\n\n")
        f.write(f"Cosine matrix:\n{cos}\n\n")
        f.write(f"off-diag mean cos = "
                f"{(cos.sum() - len(names)) / (len(names) * (len(names) - 1)):.4f}\n")
        f.write(f"PCA-8 explained variance: {pca.explained_variance_ratio_}\n")
    print(f"\n[probe] Report → {out}")


if __name__ == "__main__":
    main()
