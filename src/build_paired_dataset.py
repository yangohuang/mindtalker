"""
Stage A · 构建 paired dataset：transcripts + 16k audio → (text, real_w2v_features) pairs

输入：
  data/transcripts/*.json (ASR 转写，含 segment 时间戳)
  data/audio_16k/*.wav

输出：
  data/paired/{idx:06d}.npz   每段一个 npz 含：
    - text: str
    - bridge_states: (T_text, 768) — 待训时再过 MiniMind-O 抽（这里只存 text）
    - real_w2v_features: (T_audio, 12, 768) fp16 — 真人音频段的 wav2vec2
    - audio_path / start / end

  data/paired_index.json — list of all paired entries with metadata

设计选择：
- bridge_states 在训练 DataLoader 内部按需抽（因为要带 MiniMind-O 在 GPU；预存 GPU pickle 不灵活）
- real_w2v_features 预存 fp16 节省 50% 空间（4000 段 30s ≈ 55GB 而不是 110GB）
- segment 长度 1-10s 过滤（短的 ASR 不可靠，长的 wav2vec2 处理慢）

用法：
    PYTHONNOUSERSITE=1 /home/yg/miniforge3/envs/flashhead/bin/python src/build_paired_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from phase1_pipeline import load_wav2vec2, extract_wav2vec_features


ROOT = Path("/home/yg/yg/code/mindtalker/research")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_seg_s", type=float, default=1.0)
    parser.add_argument("--max_seg_s", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    OUT_DIR = ROOT / "data/paired"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH = ROOT / "data/paired_index.json"

    transcripts = sorted((ROOT / "data/transcripts").glob("*.json"))
    if args.limit:
        transcripts = transcripts[: args.limit]
    print(f"[paired] {len(transcripts)} transcripts found")

    print(f"[paired] Loading wav2vec2 on {args.device}")
    wav2vec2 = load_wav2vec2(args.device)

    index = []
    pair_idx = 0
    for ti, tj_path in enumerate(transcripts):
        with open(tj_path) as f:
            tj = json.load(f)
        audio_path = ROOT / tj["audio_path"]
        if not audio_path.exists():
            print(f"  [skip] {tj['name']}: audio missing")
            continue

        full_audio, sr = sf.read(str(audio_path), dtype="float32")
        if sr != 16000:
            full_audio = librosa.resample(full_audio, orig_sr=sr, target_sr=16000)
        full_audio_t = torch.from_numpy(full_audio).float()

        for seg_idx, seg in enumerate(tj["segments"]):
            s, e = seg["start"], seg["end"]
            dur = e - s
            if dur < args.min_seg_s or dur > args.max_seg_s:
                continue
            if not seg["text"].strip():
                continue

            # crop audio segment
            seg_audio = full_audio_t[int(s * 16000): int(e * 16000)]
            if len(seg_audio) < 0.5 * 16000:  # sanity check
                continue

            # extract wav2vec2 features
            with torch.no_grad():
                feats = extract_wav2vec_features(seg_audio, wav2vec2, fps=25,
                                                  device=args.device).cpu()
            if args.fp16:
                feats = feats.half()
            feats_np = feats.numpy()

            out_path = OUT_DIR / f"{pair_idx:06d}.npz"
            np.savez_compressed(
                out_path,
                text=seg["text"],
                features=feats_np,
                source_video=tj["name"],
                start_s=s,
                end_s=e,
            )
            index.append({
                "id": pair_idx,
                "video": tj["name"],
                "seg": seg_idx,
                "text": seg["text"],
                "duration_s": dur,
                "feats_shape": list(feats_np.shape),
                "path": f"data/paired/{pair_idx:06d}.npz",
            })
            pair_idx += 1

        if (ti + 1) % 50 == 0:
            print(f"  [{ti+1}/{len(transcripts)}] {pair_idx} pairs built")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    total_size = sum((ROOT / p["path"]).stat().st_size for p in index) / 1e9
    print(f"\n[paired] DONE. {pair_idx} pairs, total {total_size:.1f} GB on disk")
    print(f"[paired] Next: python src/h1_w2v_head_train_full.py")


if __name__ == "__main__":
    main()
