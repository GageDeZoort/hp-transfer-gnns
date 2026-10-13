"""LapPE encoder for SimpleGNN/SimpleGIN with concat fusion.

Implements the paper recipe:
  f_eig = (1/K) * Σ_i [ (1/sqrt(D/2)) * σ((1/(σ_0*sqrt(2))) * [v_i||λ_i] @ W_a) @ W_b ]
  f_input = (1/sqrt(n0)) * X @ W_c
  output = [f_eig || f_input]

W_a ∈ R^{2×D}, W_b ∈ R^{D×D/2}, W_c ∈ R^{n0×D/2}, all ~ N(0, σ_0²).
Sign flip is applied to eigenvectors during training.
"""

from __future__ import annotations

from math import sqrt

import torch
from torch.nn import Linear, ELU, init


class SimpleLapPEEncoder(torch.nn.Module):
    """LapPE encoder matching concat-style LapPE in the paper."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        max_freqs: int,
        sigma_0: float,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_freqs = max_freqs
        self.sigma_0 = sigma_0
        if out_dim % 2 != 0:
            raise ValueError(f"LapPE concat encoder requires even out_dim, got {out_dim}")
        self.pe_dim = out_dim // 2

        # W_a: (2, D) maps [v_i, λ_i] -> D
        self.linear_a = Linear(2, out_dim, bias=False)
        # W_b: (D, D/2) for eigen branch output.
        self.linear_b = Linear(out_dim, self.pe_dim, bias=False)
        # W_c: (n0, D/2) for input branch output.
        self.linear_c = Linear(in_dim, self.pe_dim, bias=False)

        self.act = ELU()
        self._reset_parameters()

    def _reset_parameters(self):
        std = self.sigma_0
        for m in [self.linear_a, self.linear_b, self.linear_c]:
            m.reset_parameters()
            for p in m.parameters():
                init.normal_(p.data, mean=0.0, std=std)

    def forward(
        self, x: torch.Tensor, evals: torch.Tensor, evecs: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (N, in_dim), evals: (N, k), evecs: (N, k)
        Returns: (N, out_dim)
        """
        N, k = evecs.shape
        D_half = self.pe_dim

        # Random sign flips of eigenvectors during training
        if self.training:
            sign_flip = torch.rand(k, device=evecs.device)
            sign_flip = torch.where(sign_flip >= 0.5, 1.0, -1.0)
            evecs = evecs * sign_flip.unsqueeze(0)

        # Form [v_i || lambda_i] for each node/frequency: (N, k, 2)
        pos_enc = torch.stack([evecs, evals], dim=-1)

        # Convert NaN-padded frequencies to zeros, consistent with zero-padding LapPE.
        pos_enc = torch.nan_to_num(pos_enc, nan=0.0)

        # (1 / (sigma_0 * sqrt(2))) * [v_i || lambda_i] W_a
        h = self.linear_a(pos_enc) / (self.sigma_0 * sqrt(2))  # (N, k, D)
        h = self.act(h)

        # (1/sqrt(D/2)) * sigma(...) * W_b
        h = self.linear_b(h) / sqrt(D_half)  # (N, k, D/2)

        # (1/K) sum_i (...) over frequencies.
        # Use runtime k so this remains correct if a dataset chooses a different LapPE width.
        K = max(1, k)
        f_eig = h.sum(dim=1) / K  # (N, D/2)

        # Input branch: (1/sqrt(n0)) * X W_c -> (N, D/2)
        n0 = x.shape[-1]
        f_input = self.linear_c(x) / sqrt(n0)

        # Final encoder output: [f_eig || f_input] in R^{N x D}.
        return torch.cat((f_eig, f_input), dim=-1)
