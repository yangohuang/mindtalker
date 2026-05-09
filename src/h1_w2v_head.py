"""
H1 wav2vec2 head: maps MiniMind-O bridge_states (B, T_text, 768) to
wav2vec2-style features (B, T_audio, 12, 768) at 25fps.

Architecture (A1.1 minimal):
  Linear(768 → 768) (per-frame transform)
  + 12 layer-specific heads (Linear(768 → 768) × 12)
  + temporal upsample/interpolate from T_text frame rate (text token rate)
    to T_audio = T_text * resample_factor (we use ~5 audio frames per text token)

Total params: ~7M (similar to BridgeMLPLight scale)

Note: temporal alignment is the open issue. Text token rate ≠ wav2vec2 25fps.
For spike, we use a fixed factor based on training-set average; in real H1 we'd
co-train an attention-based aligner.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class W2VHead(nn.Module):
    """A1.1 wav2vec2 head: bridge → 12-layer wav2vec2 features.

    Input:  (B, T_text, 768)   — MiniMind-O bridge_states
    Output: (B, T_audio, 12, 768)  — wav2vec2-style features at 25fps

    Strategy:
      1. per-token transform: Linear(768→768)
      2. temporal interpolate from T_text → T_audio
      3. 12-layer expansion: 12 independent Linear(768→768)
      4. residual: each output layer = transformed + scaled bridge_states
    """

    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pre = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        # 12 independent heads
        self.layer_heads = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        # learnable per-layer scale (init small so first forward ≈ residual = bridge_states)
        self.layer_residual_scale = nn.Parameter(torch.full((num_layers,), 0.5))

    def forward(self, bridge_states: torch.Tensor, target_audio_frames: int) -> torch.Tensor:
        """
        bridge_states: (B, T_text, 768)
        target_audio_frames: T_audio (e.g. audio_len_seconds * 25)
        returns: (B, T_audio, 12, 768)
        """
        B, T_text, D = bridge_states.shape

        # 1. pre-transform
        h = self.pre(bridge_states)  # (B, T_text, 768)

        # 2. temporal upsample to T_audio via linear interpolation
        # convert (B, T_text, D) → (B, D, T_text) for F.interpolate
        h_t = h.transpose(1, 2)  # (B, D, T_text)
        h_t = F.interpolate(h_t, size=target_audio_frames, mode="linear", align_corners=False)
        h = h_t.transpose(1, 2)  # (B, T_audio, 768)

        # also interpolate raw bridge_states for residual path
        b_t = bridge_states.transpose(1, 2)
        b_t = F.interpolate(b_t, size=target_audio_frames, mode="linear", align_corners=False)
        bridge_interp = b_t.transpose(1, 2)  # (B, T_audio, 768)

        # 3. 12-layer expansion + residual
        outs = []
        for l in range(self.num_layers):
            transformed = self.layer_heads[l](h)               # (B, T_audio, 768)
            mixed = transformed + self.layer_residual_scale[l] * bridge_interp
            outs.append(mixed)
        return torch.stack(outs, dim=2)  # (B, T_audio, 12, 768)


if __name__ == "__main__":
    head = W2VHead(hidden_size=768, num_layers=12)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"W2VHead params: {n_params/1e6:.2f}M")

    # smoke
    bridge = torch.randn(2, 28, 768)  # B=2, T_text=28
    out = head(bridge, target_audio_frames=104)
    print(f"input  (B, T_text, D): {tuple(bridge.shape)}")
    print(f"output (B, T_audio, L, D): {tuple(out.shape)}")
