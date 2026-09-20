"""
Stage A · ASR 转写 + VAD 切句：data/audio_16k/*.wav → data/transcripts/*.json

用 SenseVoice-Small (FunASR) — 中文优先，~5x realtime on 4090。
4000 段 × 30s audio ≈ 33h audio / 5x = 6-8h GPU。

输出 JSON 格式（每段视频一个）：
{
  "name": "video_001",
  "duration_s": 32.5,
  "segments": [
    {"start": 0.5, "end": 4.2, "text": "今天天气不错"},
    {"start": 4.5, "end": 8.1, "text": "适合出门散步"},
    ...
  ]
}

用法（在 minicpm conda env 跑，已装 funasr）：
    PYTHONNOUSERSITE=1 conda run -n minicpm python src/asr_transcribe.py
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

ROOT = PROJECT_ROOT
AUDIO_DIR = ROOT / "data/audio_16k"
TRANSCRIPT_DIR = ROOT / "data/transcripts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_path", default=str(MINIMIND_REPO / "model/SenseVoiceSmall"),
                         help="SenseVoiceSmall path; if not exists, will try to download")
    parser.add_argument("--vad_model", default="fsmn-vad",
                         help="VAD model id (FunASR built-in)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of files (for testing)")
    args = parser.parse_args()

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[asr] Loading SenseVoice-Small + VAD on {args.device}")
    from funasr import AutoModel

    # SenseVoice-Small + VAD pipeline
    model_kwargs = dict(
        model=args.model_path if Path(args.model_path).exists() else "iic/SenseVoiceSmall",
        vad_model=args.vad_model,
        vad_kwargs={"max_single_segment_time": 30000},  # 30s max per chunk
        device=args.device,
        disable_update=True,
    )
    model = AutoModel(**model_kwargs)
    print(f"[asr] Model loaded")

    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    print(f"[asr] {len(wavs)} audio files to transcribe")

    n_done = n_skip = n_fail = 0
    for i, wav in enumerate(wavs):
        name = wav.stem
        out = TRANSCRIPT_DIR / f"{name}.json"
        if out.exists():
            n_skip += 1
            continue

        try:
            res = model.generate(
                input=str(wav),
                cache={},
                language="zh",
                use_itn=True,        # inverse text normalization (numbers, punct)
                batch_size_s=60,
            )
            # res is list of dicts with 'text' field; FunASR also returns segment timestamps if VAD active
            segments = []
            for r in res:
                text = r.get("text", "").strip()
                # SenseVoice prepends emotion / event tags like <|HAPPY|><|Speech|> — strip them
                while text.startswith("<|") and "|>" in text:
                    text = text.split("|>", 1)[1]
                if text:
                    segments.append({
                        "start": float(r.get("start", 0)) / 1000,  # ms → s
                        "end": float(r.get("end", 0)) / 1000,
                        "text": text,
                    })

            with open(out, "w", encoding="utf-8") as f:
                json.dump({
                    "name": name,
                    "audio_path": str(wav.relative_to(ROOT)),
                    "n_segments": len(segments),
                    "segments": segments,
                }, f, ensure_ascii=False, indent=2)
            n_done += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(wavs)}] done={n_done} skip={n_skip} fail={n_fail}")
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL] {name}: {e}")

    print(f"\n[asr] DONE. done={n_done} skip={n_skip} fail={n_fail}")
    print(f"[asr] Next: python src/build_paired_dataset.py")


if __name__ == "__main__":
    main()
