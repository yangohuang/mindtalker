#!/usr/bin/env bash
# Stage A · 视频预处理：raw_videos/*.mp4 → 16k audio + face crops
#
# 用法（视频到来后）：
#   bash scripts/preprocess_videos.sh
#
# 输入：data/raw_videos/*.mp4 (or .webm / .mkv)
# 输出：
#   data/audio_16k/{name}.wav        16kHz mono PCM WAV
#   data/face_crops/{name}/          512x512 face crops per frame (only if needed for FlashHead retrain)
#   data/preprocess.log
#
# 性能（4090）：
#   ffmpeg audio extraction: CPU only, 4000 段约 1-2h（并行 8 进程）
#   face crop (mediapipe): 跳过此步可省 4-6h（H1 训 W2VHead 不需要 face crop，只需要 audio）

set -e
cd /home/yg/yg/code/mindtalker/research

RAW_DIR=data/raw_videos
AUDIO_DIR=data/audio_16k
FACE_DIR=data/face_crops
LOG=data/preprocess.log
mkdir -p "$AUDIO_DIR" "$FACE_DIR"

N_VIDEOS=$(find "$RAW_DIR" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" -o -name "*.mov" \) | wc -l)
echo "[preprocess] Found $N_VIDEOS videos in $RAW_DIR"

if [ "$N_VIDEOS" -eq 0 ]; then
    echo "[preprocess] No videos found. Drop your videos into $RAW_DIR/ first."
    exit 0
fi

# ---- Step 1: extract 16kHz mono audio (parallel via xargs) ----
echo "[preprocess] === Step 1: extract audio (parallel 8) ==="
find "$RAW_DIR" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" -o -name "*.mov" \) \
  | xargs -I{} -P8 bash -c '
    src="$1"
    name=$(basename "$src" | sed "s/\.[^.]*$//")
    out="data/audio_16k/${name}.wav"
    if [ ! -f "$out" ]; then
      ffmpeg -y -i "$src" -ac 1 -ar 16000 -vn -loglevel error "$out" 2>&1 || echo "FAIL: $src"
    fi
  ' _ {}

n_audio=$(ls "$AUDIO_DIR"/*.wav 2>/dev/null | wc -l)
echo "[preprocess] === Step 1 done: $n_audio audio files ==="

# Optional Step 2: face crop (skip by default — not needed for W2VHead training)
# Uncomment when needed for FlashHead fine-tune:
# echo "[preprocess] === Step 2: face crop (mediapipe) ==="
# python src/face_crop_batch.py --in_dir $RAW_DIR --out_dir $FACE_DIR --size 512

echo "[preprocess] DONE. Next: python src/asr_transcribe.py"
