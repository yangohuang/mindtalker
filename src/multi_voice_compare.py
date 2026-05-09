"""Multi-voice TTS vs real per-layer cos comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import librosa
import soundfile as sf

from phase1_pipeline import load_wav2vec2, extract_wav2vec_features


def main():
    wav2vec2 = load_wav2vec2(device="cpu")

    # real reference: podcast_sichuan
    real_feats = np.load("data/spike_001_features.npy")  # (T, 12, 768)

    voices = {
        "dylan":    "data/tts_b1_001_features.npy",  # already extracted
        "vivian":   "data/tts_vivian_16k.wav",
        "ryan":     "data/tts_ryan_16k.wav",
        "uncle_fu": "data/tts_uncle_fu_16k.wav",
    }

    print(f"{'voice':10s} {'dur(s)':>7s} | per-layer cos vs real_zh_podcast")
    print(f"{'':10s} {'':>7s} | L1     L3     L6     L9     L11    L12")
    print(f"{'-'*10} {'-'*7}-+-{'-'*42}")

    for v, src in voices.items():
        if src.endswith(".npy"):
            feats = np.load(src)
        else:
            y, sr = sf.read(src, dtype="float32")
            if y.ndim == 2: y = y.mean(axis=1)
            feats = extract_wav2vec_features(torch.from_numpy(y).float(), wav2vec2, fps=25, device="cpu").numpy()
            np.save(f"data/tts_{v}_features.npy", feats)

        cos_per_layer = []
        for l in range(12):
            f = feats[:, l, :].mean(0); f = f/(np.linalg.norm(f)+1e-8)
            r = real_feats[:, l, :].mean(0); r = r/(np.linalg.norm(r)+1e-8)
            cos_per_layer.append(float(np.dot(f, r)))

        dur = feats.shape[0] / 25
        cs = cos_per_layer
        print(f"{v:10s} {dur:7.2f} | {cs[0]:.3f}  {cs[2]:.3f}  {cs[5]:.3f}  {cs[8]:.3f}  {cs[10]:.3f}  {cs[11]:.3f}")


if __name__ == "__main__":
    main()
