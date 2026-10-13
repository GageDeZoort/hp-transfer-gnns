"""Standard-parameterization (fan-in) node encoders.

These mirror the structure of the transfer-parameterized encoders in
``deep_node_encoder.py`` and ``simple_lappe.py``, but every linear map is
initialized with variance ``1 / fan_in`` and no forward multiplier is applied.
That keeps the parameter gradients free of any ``sigma_0`` / width factors.
"""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor
from torch.nn import ELU, Linear, Module, init


def fan_in_normal_(layer: Linear) -> Linear:
    """Initialize ``layer.weight`` as N(0, 1 / fan_in) and zero any bias."""
    fan_in = layer.weight.shape[1]
    init.normal_(layer.weight, mean=0.0, std=1.0 / sqrt(fan_in))
    if layer.bias is not None:
        init.zeros_(layer.bias)
    return layer


class StandardNodeEncoder(Module):
    """Two-hidden-layer node encoder under standard (fan-in) parameterization."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = ELU()

        self.w1 = fan_in_normal_(Linear(in_dim, out_dim, bias=False))
        self.w2 = fan_in_normal_(Linear(out_dim, out_dim, bias=False))
        self.w3 = fan_in_normal_(Linear(out_dim, out_dim, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        x = self.w2(self.act(x))
        return self.w3(self.act(x))


class StandardLapPEEncoder(Module):
    """Concat-fusion LapPE encoder under standard (fan-in) parameterization."""

    def __init__(self, in_dim: int, out_dim: int, max_freqs: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_freqs = max_freqs
        if out_dim % 2 != 0:
            raise ValueError(f"LapPE concat encoder requires even out_dim, got {out_dim}")
        self.pe_dim = out_dim // 2

        self.linear_a = fan_in_normal_(Linear(2, out_dim, bias=False))
        self.linear_b = fan_in_normal_(Linear(out_dim, self.pe_dim, bias=False))
        self.linear_c = fan_in_normal_(Linear(in_dim, self.pe_dim, bias=False))
        self.act = ELU()

    def forward(self, x: Tensor, evals: Tensor, evecs: Tensor) -> Tensor:
        _, k = evecs.shape

        if self.training:
            sign_flip = torch.rand(k, device=evecs.device)
            sign_flip = torch.where(sign_flip >= 0.5, 1.0, -1.0)
            evecs = evecs * sign_flip.unsqueeze(0)

        pos_enc = torch.stack([evecs, evals], dim=-1)
        pos_enc = torch.nan_to_num(pos_enc, nan=0.0)

        h = self.act(self.linear_a(pos_enc))
        h = self.linear_b(h)

        # Average over frequencies so the encoder output is K-independent.
        f_eig = h.sum(dim=1) / max(1, k)
        f_input = self.linear_c(x)
        return torch.cat((f_eig, f_input), dim=-1)
