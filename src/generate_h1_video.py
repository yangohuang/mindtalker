"""
H1 end-to-end inference spike (random-init W2VHead).

Goal: prove the plumbing works:
    text → MiniMind-O Thinker → bridge_states → W2VHead → fake wav2vec2 features
    → FlashHead Lite → video

W2VHead is random-init (layer_heads near-identity + layer_residual_scale=1.0)
so the "fake features" are smooth interpolations of bridge_states. FlashHead
will produce SOME video — it might look weird, but the pipeline running E2E
proves H1 inference is viable on a single 4090.

Real H1 training (W2VHead supervised against real wav2vec2 features) is the
4-5 week follow-up. This spike proves the plumbing in 5 minutes.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


FLASHHEAD_ROOT = Path("/home/yg/yg/code/github/SoulX/SoulX-FlashHead")
MINIMIND_REPO = Path("/home/yg/yg/code/github/minimind-o")
WEIGHT_DIR = MINIMIND_REPO / "out"
RESEARCH_ROOT = Path("/home/yg/yg/code/mindtalker/research")


def load_minimind(device: str):
    sys.path.insert(0, str(MINIMIND_REPO))
    import importlib.machinery
    for mod_name in ("onnxruntime", "funasr"):
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
            m.AutoModel = lambda *a, **k: None
            sys.modules[mod_name] = m
    from model.model_omni import MiniMindOmni, OmniConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHT_DIR), trust_remote_code=True)
    cfg = OmniConfig()
    model = MiniMindOmni(config=cfg,
                         audio_encoder_path="/nonexistent",
                         vision_model_path="/nonexistent")
    sd = torch.load(str(WEIGHT_DIR / "pytorch_model.bin"), map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.audio_encoder = None
    model.vision_encoder = None
    model = model.half().to(device).eval()
    return model, tokenizer


def get_bridge_states(model, tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = torch.tensor(tokenizer(inputs_text).data["input_ids"],
                              dtype=torch.long, device=device).unsqueeze(0)

    captured = {}
    bridge_layer_idx = model.config.bridge_layer

    def hook_fn(_, __, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["bridge"] = h.detach()

    handle = model.thinker.layers[bridge_layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    handle.remove()
    return captured["bridge"], input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="今天天气不错，适合出门散步，我们去公园走一走吧。")
    parser.add_argument("--audio_for_timing", type=Path,
                        default=RESEARCH_ROOT / "data/tts_b1_001_16k.wav",
                        help="Real audio used only to determine output video frame count")
    parser.add_argument("--save", type=Path, default=RESEARCH_ROOT / "data/flashhead_h1_001.mp4")
    parser.add_argument("--cond_image", type=Path, default=FLASHHEAD_ROOT / "examples/girl.png")
    parser.add_argument("--ckpt_dir", type=Path, default=FLASHHEAD_ROOT / "models/SoulX-FlashHead-1_3B")
    parser.add_argument("--wav2vec_dir", type=Path, default=FLASHHEAD_ROOT / "models/wav2vec2-base-960h")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.save.parent.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load MiniMind-O + extract bridge_states ----
    print("[H1] Loading MiniMind-O...")
    minimind, tokenizer = load_minimind(args.device)
    bridge_states, input_ids = get_bridge_states(minimind, tokenizer, args.prompt, args.device)
    print(f"[H1] bridge_states shape: {tuple(bridge_states.shape)}")
    print(f"[H1] bridge_states stats: mean {bridge_states.float().mean().item():.3f} std {bridge_states.float().std().item():.3f}")

    # free up GPU mem
    del minimind
    torch.cuda.empty_cache()

    # ---- Step 2: load W2VHead (random init for spike) ----
    sys.path.insert(0, str(RESEARCH_ROOT / "src"))
    from h1_w2v_head import W2VHead
    head = W2VHead(hidden_size=768, num_layers=12).to(args.device).half().eval()
    print(f"[H1] W2VHead params: {sum(p.numel() for p in head.parameters())/1e6:.2f}M (random init)")

    # ---- Step 3: load FlashHead pipeline + monkey-patch preprocess_audio ----
    os.chdir(FLASHHEAD_ROOT)
    sys.path.insert(0, str(FLASHHEAD_ROOT))
    import generate_video
    from flash_head.inference import (
        get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline
    )
    import librosa
    import time

    pipeline = get_pipeline(world_size=1, ckpt_dir=str(args.ckpt_dir),
                            wav2vec_dir=str(args.wav2vec_dir), model_type="lite")
    get_base_data(pipeline, cond_image_path_or_dir=str(args.cond_image),
                  base_seed=42, use_face_crop=False)

    infer_params = get_infer_params()
    sr = infer_params["sample_rate"]
    fps = infer_params["tgt_fps"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num

    # determine target audio frame count from a reference audio (just for timing)
    audio_for_timing, _ = librosa.load(str(args.audio_for_timing), sr=sr, mono=True)
    target_audio_frames = int(len(audio_for_timing) / sr * fps)
    print(f"[H1] target video frames: {target_audio_frames} ({target_audio_frames/fps:.2f}s @ {fps}fps)")

    # monkey-patch: bypass wav2vec2 entirely, return W2VHead(bridge) instead
    def patched_preprocess(speech_array, sr=16000, fps=25):
        with torch.no_grad():
            n_frames = int(len(speech_array) / sr * fps)
            head_out = head(bridge_states.half(), target_audio_frames=n_frames)
            # head_out: (1, T, 12, 768) → squeeze(0) → (T, 12, 768)  matching FlashHead's expectation
            return head_out.squeeze(0).to(args.device)

    pipeline.preprocess_audio = patched_preprocess

    print("[H1] Pipeline patched: preprocess_audio = W2VHead(MiniMind-O bridge_states)")

    # ---- Step 4: pad timing audio + run FlashHead chunks ----
    human_speech_array_slice_len = slice_len * sr // fps
    human_speech_array_frame_num = frame_num * sr // fps

    remainder = (len(audio_for_timing) - human_speech_array_frame_num) % human_speech_array_slice_len
    if remainder > 0:
        pad = human_speech_array_slice_len - remainder
        audio_for_timing = np.concatenate([audio_for_timing, np.zeros(pad, dtype=audio_for_timing.dtype)])

    audio_embedding_all = get_audio_embedding(pipeline, audio_for_timing)
    print(f"[H1] audio_embedding_all shape: {tuple(audio_embedding_all.shape)}")

    n_chunks = (audio_embedding_all.shape[1] - frame_num) // slice_len + 1
    chunks = [
        audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous()
        for i in range(n_chunks)
    ]

    generated = []
    for i, chunk in enumerate(chunks):
        if chunk.shape[1] < frame_num:
            continue
        torch.cuda.synchronize()
        t = time.time()
        v = run_pipeline(pipeline, chunk)
        if i != 0:
            v = v[motion_frames_num:]
        torch.cuda.synchronize()
        print(f"[H1] chunk {i}: {time.time()-t:.2f}s")
        generated.append(v.cpu())

    save_path = str(args.save)
    generate_video.save_video(generated, save_path, str(args.audio_for_timing), fps=fps)
    print(f"[H1] DONE → {save_path}")
    print(f"[H1] NOTE: W2VHead is random init — output is plumbing proof, not trained quality")


if __name__ == "__main__":
    main()
