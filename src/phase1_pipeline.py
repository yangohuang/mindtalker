"""
Phase 1 朴素链路 baseline pipeline.

Chain: user audio → MiniCPM-o (4.5-awq INT4) → 24kHz waveform
     → resample 24k→16k
     → wav2vec2-base-960h (12 layer hidden_states, 25fps via FlashHead's linear_interpolation)
     → FlashHead Lite (32 cross-attn tokens × 1536D via AudioProjModel)
     → 25fps video frames

Goal: measure SyncNet (LSE-D / LSE-C) baseline for H0/H1 reference.

Run on flashhead conda env (torch 2.7.1+cu128) — has FlashHead deps.
For MiniCPM-o inference, use minicpm env (torch 2.6.0+cu124) — they're separate stages.
This skeleton encodes Stage 2-4 (audio → wav2vec2 → FlashHead). Stage 1 (MiniCPM-o)
is run separately and outputs cached audio waveforms.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


FLASHHEAD_ROOT = Path("/path/to/yg/code/github/SoulX/SoulX-FlashHead")
WAV2VEC_PATH = FLASHHEAD_ROOT / "models" / "wav2vec2-base-960h"
FLASHHEAD_CKPT = FLASHHEAD_ROOT / "models" / "SoulX-FlashHead-1_3B"


def load_wav2vec2(device: str = "cuda"):
    """Load FlashHead's wav2vec2 wrapper (with linear_interpolation to video fps)."""
    sys.path.insert(0, str(FLASHHEAD_ROOT))
    from flash_head.audio_analysis.wav2vec2 import Wav2Vec2Model
    from transformers import Wav2Vec2Config

    config = Wav2Vec2Config.from_pretrained(WAV2VEC_PATH)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC_PATH, config=config).to(device).eval()
    return model


@torch.no_grad()
def extract_wav2vec_features(
    audio_16k: torch.Tensor,
    wav2vec2,
    fps: int = 25,
    device: str = "cuda",
):
    """Extract 25fps × 12 layers × 768D features from 16kHz audio.

    Args:
        audio_16k: (T,) torch tensor at 16kHz
        wav2vec2: FlashHead's wrapped Wav2Vec2Model
        fps: target video fps (default 25)

    Returns:
        features: (T_frames, 12, 768)  per-frame stacked hidden_states[1:]
    """
    if audio_16k.ndim == 1:
        audio_16k = audio_16k.unsqueeze(0)  # (1, T)
    audio_16k = audio_16k.to(device)

    # FlashHead computes seq_len = audio_len * fps / sr
    seq_len = int(audio_16k.shape[-1] * fps / 16000)

    out = wav2vec2(
        input_values=audio_16k,
        seq_len=seq_len,
        output_hidden_states=True,
        return_dict=True,
    )
    # hidden_states: tuple of 13 tensors (embedding + 12 transformer layers)
    # FlashHead uses [1:] (skip embedding), each shape (B, seq_len, 768)
    feats = torch.stack(out.hidden_states[1:], dim=2)  # (B, T_frames, 12, 768)
    return feats[0].cpu()  # (T_frames, 12, 768)


def resample_24k_to_16k(audio_24k: np.ndarray) -> torch.Tensor:
    """Resample 24kHz numpy audio to 16kHz torch (librosa-based)."""
    if audio_24k.ndim == 2:
        audio_24k = audio_24k.mean(axis=0)
    audio_16k = librosa.resample(audio_24k.astype(np.float32), orig_sr=24000, target_sr=16000)
    return torch.from_numpy(audio_16k).float()


def stage_minicpm_o(text: str, audio_in_path: Path | None = None):
    """Stage 1 — call MiniCPM-o 4.5-awq for end-to-end omni inference.

    Returns: (response_text, audio_24k_np)

    NOTE: Run this in `minicpm` conda env. This skeleton has the API call only;
    actual implementation deferred to phase1_minicpmo.py to keep env separation clean.
    """
    raise NotImplementedError(
        "Run via subprocess on minicpm env. See scripts/run_minicpm_o.sh"
    )


def measure_syncnet(video_path: Path, audio_path: Path) -> dict:
    """Stage 5 — compute SyncNet LSE-D / LSE-C.

    NOTE: SyncNet inference requires a separate model (Wav2Lip-style or original SyncNet).
    Implementation deferred to integration with `syncnet_python` package once installed.
    """
    raise NotImplementedError(
        "TODO: integrate syncnet_python or third_party/syncnet. "
        "Issue #ISSUE-syncnet — to install in next iter."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True, help="16kHz WAV input")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save", type=Path, default=None, help="Save features.npy")
    args = parser.parse_args()

    print(f"[phase1] Loading wav2vec2 from {WAV2VEC_PATH}")
    wav2vec2 = load_wav2vec2(args.device)

    print(f"[phase1] Loading audio from {args.audio}")
    audio_np, sr = sf.read(str(args.audio), dtype="float32")
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
    audio = torch.from_numpy(audio_np).float()

    print(f"[phase1] Audio shape: {audio.shape}, sr: 16000")
    feats = extract_wav2vec_features(audio, wav2vec2, fps=25, device=args.device)
    print(f"[phase1] Features shape: {feats.shape}  (T_frames, 12, 768)")
    print(f"[phase1] Features stats: mean={feats.mean():.4f} std={feats.std():.4f}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save, feats.numpy())
        print(f"[phase1] Saved features → {args.save}")


if __name__ == "__main__":
    main()
