"""
Batch extract wav2vec2 features for all TTS clips in data/tts_batch/.
Stores per-clip features as .npy. Computes P(synthetic) distribution stats.

Usage: python src/batch_extract_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from phase1_pipeline import extract_wav2vec_features, load_wav2vec2  # noqa: E402


DATA = Path("data")
TTS_DIR = DATA / "tts_batch"
OUT_DIR = DATA / "tts_batch_features"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[batch] Loading wav2vec2 (CPU; GPU may be busy)")
    wav2vec2 = load_wav2vec2(device="cpu")

    wavs = sorted(TTS_DIR.rglob("*.wav"))
    print(f"[batch] Found {len(wavs)} wavs")

    all_layer12_means = []
    metadata = []

    for i, wav in enumerate(wavs):
        out_path = OUT_DIR / f"{wav.parent.name}_{wav.stem}.npy"
        if out_path.exists():
            feats = np.load(out_path)
        else:
            audio_24k, sr = sf.read(str(wav), dtype="float32")
            if audio_24k.ndim == 2:
                audio_24k = audio_24k.mean(axis=1)
            audio_16k = librosa.resample(audio_24k, orig_sr=sr, target_sr=16000)
            audio = torch.from_numpy(audio_16k).float()
            feats = extract_wav2vec_features(audio, wav2vec2, fps=25, device="cpu").numpy()
            np.save(out_path, feats)

        all_layer12_means.append(feats[:, -1, :].mean(0))
        metadata.append({
            "voice": wav.parent.name,
            "clip_id": wav.stem,
            "T": int(feats.shape[0]),
            "duration_s": float(feats.shape[0] / 25.0),
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(wavs)}]")

    P_syn = np.stack(all_layer12_means)  # (N, 768)
    print(f"\n[batch] P(syn) shape: {P_syn.shape}")
    print(f"[batch] mean per dim: mean={P_syn.mean():.4f}, std={P_syn.std():.4f}")

    # Compare with P(real): podcast_sichuan and other real spike features
    real_features = []
    for f in DATA.glob("spike_*_features.npy"):
        if "tts" in f.name:
            continue
        feats = np.load(f)
        real_features.append(feats[:, -1, :].mean(0))
    P_real = np.stack(real_features)

    # Cross-distribution layer-12 cosine similarity
    P_syn_n = P_syn / (np.linalg.norm(P_syn, axis=1, keepdims=True) + 1e-8)
    P_real_n = P_real / (np.linalg.norm(P_real, axis=1, keepdims=True) + 1e-8)

    cos_intra_syn = (P_syn_n @ P_syn_n.T)
    cos_intra_real = (P_real_n @ P_real_n.T)
    cos_inter = (P_syn_n @ P_real_n.T)

    n_syn = P_syn.shape[0]
    n_real = P_real.shape[0]

    intra_syn = (cos_intra_syn.sum() - n_syn) / (n_syn * (n_syn - 1))
    intra_real = (cos_intra_real.sum() - n_real) / (n_real * (n_real - 1))
    inter = cos_inter.mean()

    print(f"\n=== Distribution analysis (layer-12, cosine) ===")
    print(f"  intra-synthetic mean cos: {intra_syn:.4f}  (N={n_syn})")
    print(f"  intra-real      mean cos: {intra_real:.4f}  (N={n_real})")
    print(f"  inter (syn↔real) mean cos: {inter:.4f}")
    print(f"\n  domain shift ≈ intra_syn - inter = {intra_syn - inter:.4f}")
    print(f"  domain shift ≈ intra_real - inter = {intra_real - inter:.4f}")

    # Per-layer analysis (the headline finding)
    print(f"\n=== Per-layer cos: mean(syn) vs mean(real) ===")
    for l in range(12):
        # mean over all syn / real clips per layer
        syn_mean_l = np.stack([np.load(OUT_DIR / f"{m['voice']}_{m['clip_id']}.npy")[:, l, :].mean(0)
                                for m in metadata]).mean(0)
        real_mean_l = np.stack([np.load(f)[:, l, :].mean(0)
                                 for f in DATA.glob("spike_*_features.npy") if "tts" not in f.name]).mean(0)
        a = syn_mean_l / (np.linalg.norm(syn_mean_l) + 1e-8)
        b = real_mean_l / (np.linalg.norm(real_mean_l) + 1e-8)
        print(f"  layer {l+1:2d}: cos = {float(np.dot(a,b)):.4f}")

    # Save P(syn), P(real) and metadata for later BridgeMLP training
    np.save(DATA / "P_syn_layer12_means.npy", P_syn)
    np.save(DATA / "P_real_layer12_means.npy", P_real)
    print(f"\n[batch] Saved P_syn / P_real layer-12 means → data/")


if __name__ == "__main__":
    main()
