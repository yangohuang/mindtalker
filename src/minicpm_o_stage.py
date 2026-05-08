"""
MiniCPM-o Stage 1 — run on `minicpm` conda env, NOT flashhead env.

Usage (from any env):
    /path/to/miniforge3/envs/minicpm/bin/python src/minicpm_o_stage.py \\
        --prompt "Tell me a short story" \\
        --out_audio data/minicpm_out_001.wav

Outputs 24kHz mono WAV via MiniCPM-o-4_5-awq's audio synthesis path.
Run in background — GPU intensive (~12GB).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


MODEL_ID = "openbmb/MiniCPM-o-4_5-awq"


def run(prompt: str, out_audio: Path, sys_prompt: str = "Reply briefly within 30 words."):
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"[minicpm-o] Loading {MODEL_ID} (will use ~12GB GPU)")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.init_tts()

    print(f"[minicpm-o] Prompt: {prompt}")
    msgs = [{"role": "user", "content": [prompt]}]
    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
        generate_audio=True,
        output_audio_path=str(out_audio),
    )
    print(f"[minicpm-o] Response: {res}")
    print(f"[minicpm-o] Audio → {out_audio}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out_audio", type=Path, required=True)
    parser.add_argument("--sys_prompt", default="Reply briefly within 30 words.")
    args = parser.parse_args()
    args.out_audio.parent.mkdir(parents=True, exist_ok=True)
    run(args.prompt, args.out_audio, args.sys_prompt)


if __name__ == "__main__":
    main()
