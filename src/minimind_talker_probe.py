"""
H1 first spike: load MiniMind-O, run text-only forward, extract bridge_states.

Goal: prove we can get the Thinker bridge_layer hidden state out of MiniMind-O
on a single 4090 in <30 seconds — this is the input to A1.1 multi-head Talker.

Usage:
    python src/minimind_talker_probe.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .project_paths import MINIMIND_REPO
else:
    from project_paths import MINIMIND_REPO

import torch


WEIGHT_DIR = MINIMIND_REPO / "out"   # downloaded by huggingface-cli
sys.path.insert(0, str(MINIMIND_REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="今天天气不错，适合出门散步。")
    parser.add_argument("--save", type=Path, default=Path("data/minimind_bridge_001.pt"))
    args = parser.parse_args()

    # ---- Load model directly (bypass transformers check_imports for funasr/onnxruntime) ----
    from transformers import AutoTokenizer

    print(f"[probe] Loading MiniMind-O from {WEIGHT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHT_DIR), trust_remote_code=True)

    # Monkey-patch missing modules so model_omni imports cleanly
    import types
    import importlib.machinery
    for mod_name in ("onnxruntime", "funasr"):
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
            m.AutoModel = lambda *a, **k: None
            sys.modules[mod_name] = m

    # MINIMIND_REPO already in sys.path; import as package so relative imports work
    from model.model_omni import MiniMindOmni, OmniConfig

    cfg = OmniConfig()
    print(f"[probe] OmniConfig defaults loaded")

    model = MiniMindOmni(
        config=cfg,
        audio_encoder_path="/nonexistent",   # → load_sensevoice will warn + return None
        vision_model_path="/nonexistent",    # → load_vision will warn + return None
    )
    state_dict = torch.load(str(WEIGHT_DIR / "pytorch_model.bin"),
                            map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[probe] missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    if missing[:5]:
        print(f"  missing samples: {missing[:5]}")
    if unexpected[:5]:
        print(f"  unexpected samples: {unexpected[:5]}")

    model.audio_encoder = None
    model.vision_encoder = None
    model = model.half().to(args.device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[probe] Model params: {n_params/1e6:.2f}M")
    print(f"[probe] config: hidden_size={model.config.hidden_size}, "
          f"num_hidden_layers={model.config.num_hidden_layers}, "
          f"bridge_layer={model.config.bridge_layer}, "
          f"talker_hidden_size={model.config.talker_hidden_size}, "
          f"num_talker_hidden_layers={model.config.num_talker_hidden_layers}")

    # ---- Encode text prompt ----
    print(f"[probe] Prompt: {args.prompt}")
    messages = [{"role": "user", "content": args.prompt}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"[probe] Templated text: {inputs_text[:200]}...")
    input_ids = torch.tensor(tokenizer(inputs_text).data["input_ids"],
                              dtype=torch.long, device=args.device).unsqueeze(0)
    print(f"[probe] input_ids shape: {tuple(input_ids.shape)}")

    # ---- Hook bridge_states ----
    captured = {}

    # Find Thinker layers and hook layer at bridge_layer
    bridge_layer_idx = model.config.bridge_layer
    target_layer = model.thinker.layers[bridge_layer_idx]

    def hook_fn(module, inputs, output):
        # output may be (hidden_states, present_kv) tuple
        h = output[0] if isinstance(output, tuple) else output
        captured["bridge"] = h.detach()
        print(f"[probe] [hook] bridge_layer={bridge_layer_idx} captured: {tuple(h.shape)}")

    handle = target_layer.register_forward_hook(hook_fn)

    # ---- Run forward (text-only, no audio_inputs / pixel_values) ----
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids)
        except Exception as e:
            print(f"[probe] Forward failed: {e}")
            print("[probe] Trying with attention_mask=None and other minimal args...")
            out = model.forward(input_ids=input_ids, attention_mask=None,
                                past_key_values=None, use_cache=False)

    print(f"[probe] Forward output type: {type(out).__name__}")
    if hasattr(out, "logits"):
        print(f"[probe] logits shape: {tuple(out.logits.shape)}")

    handle.remove()

    bridge_states = captured.get("bridge")
    if bridge_states is None:
        print("[probe] ERROR: no bridge_states captured")
        return

    print(f"\n[probe] === bridge_states extracted ===")
    print(f"[probe] shape: {tuple(bridge_states.shape)}  (B, T, hidden_size)")
    print(f"[probe] dtype: {bridge_states.dtype}")
    print(f"[probe] mean={bridge_states.mean().item():.4f} std={bridge_states.std().item():.4f}")

    # save for downstream A1.1 head training
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "bridge_states": bridge_states.cpu(),
        "prompt": args.prompt,
        "input_ids": input_ids.cpu(),
        "config_summary": {
            "hidden_size": model.config.hidden_size,
            "bridge_layer": model.config.bridge_layer,
            "num_hidden_layers": model.config.num_hidden_layers,
            "talker_hidden_size": model.config.talker_hidden_size,
        },
    }, str(args.save))
    print(f"[probe] Saved → {args.save}")


if __name__ == "__main__":
    main()
