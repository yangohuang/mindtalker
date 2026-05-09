"""
Generate B2 video: TTS audio → wav2vec2 → BridgeMLP → FlashHead Lite → video.

Strategy: monkey-patch FlashHead pipeline's preprocess_audio() to insert
BridgeMLP between wav2vec2 hidden_states extraction and downstream rendering.

Usage:
    python src/generate_b2_video.py \\
        --audio data/tts_b1_001_16k.wav \\
        --bridge_ckpt experiments/H0-bridge-mlp-baseline/results/bridge_mlp_unpaired.pt \\
        --save data/flashhead_b2_001.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange


FLASHHEAD_ROOT = Path("/path/to/yg/code/github/SoulX/SoulX-FlashHead")
RESEARCH_ROOT = Path("/path/to/yg/code/mindtalker/research")
sys.path.insert(0, str(FLASHHEAD_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT))

# --- import generate_video.py utilities (subprocess fallback if can't) ---


def patch_pipeline_with_bridge(pipeline, bridge_ckpt: Path, device: str = "cuda"):
    """Monkey-patch pipeline.preprocess_audio to insert BridgeMLPLight."""
    from src.bridge_mlp import BridgeMLPLight

    bridge = BridgeMLPLight(num_layers=12, dim=768, residual=True, layer_gate=True)
    bridge.load_state_dict(torch.load(str(bridge_ckpt), map_location="cpu"))
    bridge = bridge.to(device).eval()
    print(f"[B2] Loaded BridgeMLPLight from {bridge_ckpt}")
    print(f"[B2] gate values per layer: {[f'{torch.sigmoid(bridge.gate[l]).item():.3f}' for l in range(12)]}")

    orig_preprocess = pipeline.preprocess_audio

    def patched_preprocess(speech_array, sr=16000, fps=25):
        # Run original preprocess to get (T, 12, 768) audio_emb
        audio_emb = orig_preprocess(speech_array, sr=sr, fps=fps)  # (T, 12, 768)
        if audio_emb is None:
            return None

        # Bridge MLP expects (B, T, 12, 768)
        x = audio_emb.unsqueeze(0).to(device)              # (1, T, 12, 768)
        with torch.no_grad():
            x_bridged = bridge(x)                          # (1, T, 12, 768)
        out = x_bridged.squeeze(0).to(audio_emb.dtype).to(audio_emb.device)
        # Compare changes for sanity
        delta = float((out - audio_emb).abs().mean())
        print(f"[B2] BridgeMLP applied: input shape {tuple(audio_emb.shape)}, mean |Δ| {delta:.5f}")
        return out

    pipeline.preprocess_audio = patched_preprocess
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--bridge_ckpt", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--cond_image", type=Path, default=FLASHHEAD_ROOT / "examples/girl.png")
    parser.add_argument("--ckpt_dir", type=Path, default=FLASHHEAD_ROOT / "models/SoulX-FlashHead-1_3B")
    parser.add_argument("--wav2vec_dir", type=Path, default=FLASHHEAD_ROOT / "models/wav2vec2-base-960h")
    parser.add_argument("--model_type", default="lite")
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    args.save.parent.mkdir(parents=True, exist_ok=True)

    # ---- Import FlashHead's generate_video machinery ----
    os.chdir(FLASHHEAD_ROOT)  # generate_video.py uses relative imports
    import generate_video
    from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline
    import librosa
    import time
    from collections import deque

    device = "cuda"
    pipeline = get_pipeline(world_size=1, ckpt_dir=str(args.ckpt_dir), wav2vec_dir=str(args.wav2vec_dir),
                            model_type=args.model_type)
    get_base_data(pipeline, cond_image_path_or_dir=str(args.cond_image),
                  base_seed=args.base_seed, use_face_crop=False)

    # ---- INSERT BRIDGE MLP ----
    pipeline = patch_pipeline_with_bridge(pipeline, args.bridge_ckpt, device=device)

    infer_params = get_infer_params()
    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num

    human_speech_array_all, _ = librosa.load(str(args.audio), sr=sample_rate, mono=True)
    print(f"[B2] Loaded audio: {len(human_speech_array_all)/sample_rate:.2f}s")

    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    human_speech_array_frame_num = frame_num * sample_rate // tgt_fps

    # pad
    remainder = (len(human_speech_array_all) - human_speech_array_frame_num) % human_speech_array_slice_len
    if remainder > 0:
        pad_length = human_speech_array_slice_len - remainder
        human_speech_array_all = np.concatenate(
            [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
        )

    audio_embedding_all = get_audio_embedding(pipeline, human_speech_array_all)
    print(f"[B2] audio_embedding_all shape: {tuple(audio_embedding_all.shape)}")

    audio_embedding_chunks = [
        audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous()
        for i in range((audio_embedding_all.shape[1] - frame_num) // slice_len + 1)
    ]

    generated_list = []
    for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks):
        if audio_embedding_chunk.shape[1] < frame_num:
            continue
        torch.cuda.synchronize()
        start = time.time()
        video = run_pipeline(pipeline, audio_embedding_chunk)
        if chunk_idx != 0:
            video = video[motion_frames_num:]
        torch.cuda.synchronize()
        print(f"[B2] chunk {chunk_idx}: {time.time()-start:.2f}s")
        generated_list.append(video.cpu())

    # save (reuse generate_video.save_video)
    os.chdir(FLASHHEAD_ROOT)
    save_path = str(args.save)
    generate_video.save_video(generated_list, save_path, str(args.audio), fps=tgt_fps)
    print(f"[B2] DONE → {save_path}")


if __name__ == "__main__":
    main()
