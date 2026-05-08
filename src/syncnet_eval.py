"""
SyncNet evaluation interface.

Real implementation deferred (network/pip issues — flashhead env's pip resolves to ~/.local
which CLAUDE.md flags as contamination source). For now this provides:
  1. Stable interface accepting (video_path, audio_path) → dict of metrics
  2. Three back-ends:
     - 'stub'      : returns deterministic dummy values (for pipeline plumbing tests)
     - 'librosa'   : lightweight audio-feature-only sanity proxy (no real lip-sync)
     - 'syncnet'   : real SyncNet model, install deferred

Once the SyncNet model is installed (plan: clone joonson/syncnet_python or use latentsync's
sync_score module on a clean conda env), swap backend='syncnet'.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np


def _stub_metrics(video_path: Path, audio_path: Path) -> dict:
    """Deterministic dummy metrics derived from path hash. Useful only for plumbing tests."""
    h = hashlib.md5(f"{video_path}|{audio_path}".encode()).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.default_rng(seed)
    return {
        "lse_d": float(rng.uniform(7.0, 12.0)),   # FlashHead training corpus typical
        "lse_c": float(rng.uniform(5.0, 9.0)),
        "confidence": float(rng.uniform(0.3, 0.9)),
        "backend": "stub",
    }


def _librosa_proxy(video_path: Path, audio_path: Path) -> dict:
    """Naive audio-side sanity check: extract MFCC + energy + voicing.

    NOT a real lip-sync metric. Returns zeros for lse_d/lse_c (do not use for H0/H1
    evaluation). Useful only to verify val-set data integrity.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    energy = float(np.mean(y ** 2))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return {
        "lse_d": 0.0,
        "lse_c": 0.0,
        "confidence": 0.0,
        "backend": "librosa-proxy",
        "audio_energy": energy,
        "mfcc_mean": float(np.mean(mfcc)),
        "audio_seconds": float(len(y) / sr),
    }


def _real_syncnet(video_path: Path, audio_path: Path) -> dict:
    """Real SyncNet — TODO. Install path:

    Option A (recommended): on a clean conda env with PYTHONNOUSERSITE=1,
        git clone https://github.com/joonson/syncnet_python
        download SyncNet pretrained weights
        cd syncnet_python && python run_syncnet.py --videofile {video} --reference {ref}

    Option B (lightweight): use LatentSync's sync_score module
        https://github.com/bytedance/LatentSync/tree/main/sync_score

    Returns: {'lse_d': float, 'lse_c': float, 'confidence': float, 'backend': 'syncnet'}
    """
    raise NotImplementedError(
        "Real SyncNet not yet installed. See docstring for install plan."
    )


def evaluate(
    video_path: str | Path,
    audio_path: str | Path,
    backend: Literal["stub", "librosa", "syncnet"] = "stub",
) -> dict:
    video_path, audio_path = Path(video_path), Path(audio_path)

    if backend == "stub":
        return _stub_metrics(video_path, audio_path)
    elif backend == "librosa":
        return _librosa_proxy(video_path, audio_path)
    elif backend == "syncnet":
        return _real_syncnet(video_path, audio_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def evaluate_batch(
    pairs: list[tuple[str | Path, str | Path]],
    backend: Literal["stub", "librosa", "syncnet"] = "stub",
) -> dict:
    """Aggregate over (video, audio) pairs."""
    results = [evaluate(v, a, backend=backend) for v, a in pairs]
    keys = ["lse_d", "lse_c", "confidence"]
    agg = {f"{k}_mean": float(np.mean([r[k] for r in results])) for k in keys}
    agg["n"] = len(results)
    agg["backend"] = backend
    return agg


if __name__ == "__main__":
    # smoke test
    sample_audio = Path(
        "/path/to/yg/code/github/SoulX/SoulX-FlashHead/examples/podcast_sichuan_16k.wav"
    )
    print("stub:", evaluate("dummy.mp4", sample_audio, backend="stub"))
    print("librosa:", evaluate("dummy.mp4", sample_audio, backend="librosa"))
