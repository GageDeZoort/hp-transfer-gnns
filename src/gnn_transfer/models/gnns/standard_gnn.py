"""Standard-parameterization residual GCN baseline (``gcn-sp``).

Architecturally identical to :class:`~gnn_transfer.models.gnns.simple_transfer_gnn.SimpleGNN`
(same encoder/decoder options, same MPNN + 4x-expansion MLP residual blocks, same
message-passing operator, same activations, same pooling, biases off), but with
none of the CompleteP/muP machinery:

* residual branches are scaled by ``L^{-1/2}`` instead of ``L^{-1}``;
* every linear map is initialized with variance ``1 / fan_in`` and used with no
  forward multiplier, so no ``sigma_0``, ``sigma_{L+1}``, ``D`` or ``L`` factor
  ever reaches a parameter gradient;
* the optimizer sees the configured learning rate and weight decay verbatim.
"""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor
from torch.nn import ELU, LayerNorm, Linear, Module, ModuleList
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import scatter

from gnn_transfer.models.encoders.standard_encoders import (
    StandardLapPEEncoder,
    StandardNodeEncoder,
    fan_in_normal_,
)
from gnn_transfer.models.gnns.gcn import ADJACENCY_TYPES, build_mp_operator


class StandardGCN(Module):
    """Residual GCN under ordinary fan-in (standard) parameterization."""

    # Consumed by experiment scripts to skip transfer-specific optimizer rules.
    standard_parameterization = True

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        heads: int = 1,
        res_agg: float = 1.0,
        t: float | None = None,
        gamma: float = 1.0,
        optimizer: str = "adam",
        cached: bool = False,
        pool: bool = False,
        log_batch_stats: bool = False,
        encoder: str = "linear",
        decoder: str = "linear",
        adjacency: str = "normalized",
        max_freqs: int = 0,
        layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__()

        sigma_output = kwargs.pop("sigma_output", "1")
        if str(sigma_output).lower() != "1":
            raise ValueError(
                "gcn-sp uses standard parameterization and does not support "
                f"sigma_output={sigma_output!r}; only '1' is valid."
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.heads = heads
        self.D = width * heads
        if t is not None:
            res_agg = t
        self.res_agg = res_agg
        self.gamma = gamma
        self.pool = pool
        self.cached = cached
        self._cached_edge_index = None
        self.log_batch_stats = log_batch_stats
        self._last_x = None
        self.encoder_type = encoder.lower()
        self.decoder_type = decoder.lower()
        if self.encoder_type not in ("linear", "lappe", "deeper"):
            raise ValueError("encoder must be 'linear', 'lappe', or 'deeper'")
        if self.decoder_type not in ("linear", "deep", "deeper"):
            raise ValueError("decoder must be 'linear', 'deep', or 'deeper'")
        self.adjacency_type = adjacency.lower()
        if self.adjacency_type not in ADJACENCY_TYPES:
            raise ValueError(f"adjacency must be one of {ADJACENCY_TYPES}")
        self.max_freqs = max_freqs
        self.optimizer = optimizer.lower()

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = LayerNorm(self.D, bias=False, elementwise_affine=False)

        # Residual strength: L^{-1/2}, the defining depth scaling of this baseline.
        self.res_scale = 1.0 / sqrt(self.depth)

        if self.encoder_type == "lappe":
            self.embed = StandardLapPEEncoder(
                in_dim=in_dim, out_dim=self.D, max_freqs=max_freqs
            )
        elif self.encoder_type == "deeper":
            self.embed = StandardNodeEncoder(in_dim, self.D)
        else:
            self.embed = fan_in_normal_(Linear(in_dim, self.D, bias=False))

        # Residual stream: MPNN + MLP blocks, all fan-in initialized.
        self.gnn_layers = ModuleList(
            [
                fan_in_normal_(Linear(self.D, self.D, bias=False))
                for _ in range(self.depth)
            ]
        )
        self.mlp_in = ModuleList(
            [
                fan_in_normal_(Linear(self.D, 4 * self.D, bias=False))
                for _ in range(self.depth)
            ]
        )
        self.mlp_out = ModuleList(
            [
                fan_in_normal_(Linear(4 * self.D, self.D, bias=False))
                for _ in range(self.depth)
            ]
        )
        self.act = ELU()

        if self.decoder_type == "linear":
            self.out_proj = fan_in_normal_(Linear(self.D, out_dim, bias=False))
            self.dec_lin1 = None
            self.dec_lin2 = None
            self.dec_lin3 = None
        elif self.decoder_type == "deep":
            self.out_proj = None
            self.dec_lin1 = fan_in_normal_(Linear(self.D, self.D, bias=False))
            self.dec_lin2 = fan_in_normal_(Linear(self.D, out_dim, bias=False))
            self.dec_lin3 = None
        else:
            self.out_proj = None
            self.dec_lin1 = fan_in_normal_(Linear(self.D, self.D, bias=False))
            self.dec_lin2 = fan_in_normal_(Linear(self.D, self.D, bias=False))
            self.dec_lin3 = fan_in_normal_(Linear(self.D, out_dim, bias=False))

    def get_lr(self, eta_0: float, corr: float = 1.0) -> float:
        """The configured LR, verbatim: no width, depth, or sparse-input correction."""
        return float(eta_0)

    def get_weight_decay(self, lambda_0: float) -> float:
        """The configured weight decay, verbatim: no ``sqrt(D)`` rescaling."""
        return float(lambda_0)

    def make_param_groups(
        self,
        eta_0: float,
        corr: float = 1.0,
        weight_decay: float = 0.0,
        first_layer_corr: float = 1.0,
    ):
        if first_layer_corr != 1.0:
            raise ValueError(
                "gcn-sp does not support first_layer_corr != 1.0; per-layer learning-"
                "rate corrections are a hyperparameter-transfer feature."
            )
        lr = self.get_lr(eta_0)

        def split_decay(params):
            decayed, no_decayed = [], []
            for p in params:
                if p.requires_grad:
                    (no_decayed if p.ndim == 1 else decayed).append(p)
            groups = []
            if decayed:
                groups.append({"params": decayed, "weight_decay": weight_decay})
            if no_decayed:
                groups.append({"params": no_decayed, "weight_decay": 0.0})
            return groups

        if self.decoder_type == "linear":
            last_params = list(self.out_proj.parameters())
        elif self.decoder_type == "deep":
            last_params = list(self.dec_lin1.parameters()) + list(
                self.dec_lin2.parameters()
            )
        else:
            last_params = (
                list(self.dec_lin1.parameters())
                + list(self.dec_lin2.parameters())
                + list(self.dec_lin3.parameters())
            )

        param_groups = []
        for name, params in [
            ("first", list(self.embed.parameters())),
            (
                "resid",
                list(self.gnn_layers.parameters())
                + list(self.mlp_in.parameters())
                + list(self.mlp_out.parameters()),
            ),
            ("last", last_params),
        ]:
            for g in split_decay(params):
                g.update({"lr": lr, "name": name})
                param_groups.append(g)
        return param_groups

    def forward(self, data) -> Tensor:
        if self.encoder_type == "lappe":
            if not (hasattr(data, "evals") and hasattr(data, "evecs")):
                raise ValueError(
                    "encoder='lappe' requires data with evals and evecs "
                    "(use a dataset with LapPE=True, e.g. Pascal with --encoder lappe)"
                )
            x = self.embed(data.x, data.evals, data.evecs)
        else:
            x = self.embed(data.x)
        edge_index, edge_weight = self._get_edge_weight(x, data.edge_index)
        row, col = edge_index[0], edge_index[1]

        for idx, layer in enumerate(self.gnn_layers):
            h_in = self.ln(x) if self.layer_norm else x
            h = layer(self.act(h_in))
            msg = h[row] if edge_weight is None else edge_weight.view(-1, 1) * h[row]
            h = scatter(msg, col, dim=0, dim_size=x.size(0), reduce="sum")
            x_tilde = x + (self.res_scale / self.gamma) * h

            m_in = self.ln(x_tilde) if self.layer_norm else x_tilde
            mlp_h = self.mlp_in[idx](self.act(m_in))
            mlp_h = self.mlp_out[idx](self.act(mlp_h))
            x = x_tilde + self.res_scale * mlp_h

        x_nodes = x
        if self.log_batch_stats:
            x_nodes.retain_grad()
            self._last_x = x_nodes

        if self.pool:
            x = global_mean_pool(x_nodes, batch=getattr(data, "batch", None))
        else:
            x = x_nodes

        if self.decoder_type == "linear":
            return self.out_proj(x)
        if self.decoder_type == "deep":
            h = self.act(self.dec_lin1(self.act(x)))
            return self.dec_lin2(h)
        h1 = self.act(self.dec_lin1(self.act(x)))
        h2 = self.act(self.dec_lin2(h1))
        return self.dec_lin3(h2)

    def _get_edge_weight(self, x: Tensor, edge_index: Tensor):
        if self.cached and self._cached_edge_index is not None:
            return self._cached_edge_index
        edge_index, edge_weight = build_mp_operator(
            edge_index,
            x.size(0),
            self.adjacency_type,
            self.res_agg,
            x.dtype,
        )
        if self.cached:
            self._cached_edge_index = (edge_index, edge_weight)
        return edge_index, edge_weight

    def get_batch_stats(self) -> dict[str, float]:
        if self._last_x is None:
            raise RuntimeError("No batch stats available. Call forward() first.")
        x = self._last_x
        feat_fro = float(torch.linalg.norm(x).item())
        if x.grad is None:
            grad_node_norm_mean = float("nan")
        else:
            grad_node_norm_mean = float(x.grad.norm(dim=1).mean().item())
        return {
            "feat_fro": feat_fro,
            "grad_node_norm_mean": grad_node_norm_mean,
        }
