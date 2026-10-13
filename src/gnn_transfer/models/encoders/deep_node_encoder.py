"""Two-hidden-layer node encoder (W1, W2, W3) with sigma_0 init and scaling."""

from __future__ import annotations

from math import sqrt

from torch import Tensor
from torch.nn import ELU, Linear, Module, init


class DeepNodeEncoder(Module):
    """W1, W2, W3 node encoder with two hidden layers (mirrors deeper decoder depth)."""

    def __init__(self, in_dim: int, out_dim: int, sigma_0: float):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_0 = sigma_0
        self.act = ELU()

        self.w1 = Linear(in_dim, out_dim, bias=False)
        self.w2 = Linear(out_dim, out_dim, bias=False)
        self.w3 = Linear(out_dim, out_dim, bias=False)

        init.normal_(self.w1.weight, mean=0.0, std=sigma_0)
        for w in (self.w2, self.w3):
            init.normal_(w.weight, mean=0.0, std=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x) / (sqrt(self.in_dim) * self.sigma_0)
        x = self.w2(self.act(x)) / (sqrt(self.out_dim))
        return self.w3(self.act(x)) / (sqrt(self.out_dim))
