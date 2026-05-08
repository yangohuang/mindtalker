"""
Bridge MLP for H0 baseline.

Lightweight ~7M variant: per-layer Linear + cross-layer 1×1 conv.
Heavy ~75M variant: flatten 12*768 → MLP → reshape (fallback).

Frozen by upstream wav2vec2-base-960h, target: FlashHead-preferred feature distribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BridgeMLPLight(nn.Module):
    """~7M parameters. Per-layer Linear(768→768) + 1×1 conv across 12 layers.

    Input  : (B, T, 12, 768) — wav2vec2 hidden_states[1:] @ 25fps
    Output : (B, T, 12, 768) — same shape, distribution-shifted
    """

    def __init__(self, num_layers: int = 12, dim: int = 768, residual: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.residual = residual

        # per-layer transform
        self.per_layer = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )
        # cross-layer 1×1 conv (learns to mix info across wav2vec2 layers)
        self.cross_layer = nn.Conv1d(
            in_channels=num_layers,
            out_channels=num_layers,
            kernel_size=1,
            groups=1,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, L=12, D=768)
        residual = x

        # per-layer: apply Linear independently to each of the 12 layers
        out = torch.stack(
            [self.per_layer[l](x[:, :, l, :]) for l in range(self.num_layers)],
            dim=2,
        )  # (B, T, 12, 768)
        out = self.act(out)
        out = self.dropout(out)

        # cross-layer mixing: treat 12 layers as channel dim of 1d conv over feature axis
        B, T, L, D = out.shape
        out = out.permute(0, 2, 1, 3).reshape(B, L, T * D)  # (B, L, T*D)
        out = self.cross_layer(out)                          # (B, L, T*D)
        out = out.reshape(B, L, T, D).permute(0, 2, 1, 3)    # (B, T, L, D)

        if self.residual:
            out = out + residual
        return out

    @torch.no_grad()
    def init_as_identity(self):
        """Bias init: as close to identity at start (helps warmup)."""
        for lin in self.per_layer:
            nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.dirac_(self.cross_layer.weight)
        nn.init.zeros_(self.cross_layer.bias)


class BridgeMLPHeavy(nn.Module):
    """~75M parameters. Full flatten → MLP → reshape. Use as fallback if Light underfits."""

    def __init__(self, num_layers: int = 12, dim: int = 768, hidden: int = 4096):
        super().__init__()
        flat = num_layers * dim  # 9216
        self.num_layers = num_layers
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, flat),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, L, D)
        B, T, L, D = x.shape
        flat = x.view(B, T, L * D)
        out = self.net(flat) + flat
        return out.view(B, T, L, D)


if __name__ == "__main__":
    # smoke test
    light = BridgeMLPLight()
    n_params = sum(p.numel() for p in light.parameters())
    print(f"BridgeMLPLight params: {n_params/1e6:.2f}M")

    heavy = BridgeMLPHeavy()
    n_params = sum(p.numel() for p in heavy.parameters())
    print(f"BridgeMLPHeavy params: {n_params/1e6:.2f}M")

    x = torch.randn(2, 33, 12, 768)
    print("light output:", light(x).shape)
    print("heavy output:", heavy(x).shape)
