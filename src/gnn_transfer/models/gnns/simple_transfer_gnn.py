from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor
from torch.nn import ELU, LayerNorm, Linear, Module, ModuleList, init
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import scatter

from gnn_transfer.models.encoders.deep_node_encoder import DeepNodeEncoder
from gnn_transfer.models.encoders.simple_lappe import SimpleLapPEEncoder
from gnn_transfer.models.gnns.gcn import ADJACENCY_TYPES, build_mp_operator


class SimpleGNN(Module):
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
        sigma_output: str = "1",
        **kwargs,
    ):
        super().__init__()

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
            raise ValueError("adjacency must be 'normalized' or 'unnormalized'")
        self.max_freqs = max_freqs

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = LayerNorm(self.D, bias=False, elementwise_affine=False)

        sigma_output = sigma_output.lower()
        if sigma_output not in ("1", "inv-sqrt-width"):
            raise ValueError(
                "sigma_output must be '1' or 'inv-sqrt-width', "
                f"got {sigma_output!r}"
            )
        self.sigma_output = sigma_output

        self.optimizer = optimizer.lower()
        if "adam" in self.optimizer:
            sigma_0 = 1 / sqrt(self.D)
            sigma_Lp1 = 1.0 if sigma_output == "1" else 1 / sqrt(self.D)
        else:
            if sigma_output == "inv-sqrt-width":
                raise ValueError(
                    "sigma_output='inv-sqrt-width' requires an Adam-family optimizer"
                )
            sigma_0 = sqrt(self.depth)
            sigma_Lp1 = sqrt(self.depth)
        self.sigma_0 = sigma_0
        self.sigma_Lp1 = sigma_Lp1

        if self.encoder_type == "lappe":
            self.embed = SimpleLapPEEncoder(
                in_dim=in_dim,
                out_dim=self.D,
                max_freqs=max_freqs,
                sigma_0=sigma_0,
            )
        elif self.encoder_type == "deeper":
            self.embed = DeepNodeEncoder(in_dim, self.D, sigma_0=sigma_0)
        else:
            # X^(0) = (1 / (sigma_0 * sqrt(n0))) X W^(0)
            self.embed = Linear(in_dim, self.D, bias=False)
            init.normal_(self.embed.weight, mean=0.0, std=sigma_0)

        # Residual stream: GNN + MLP blocks.
        self.gnn_layers = ModuleList(
            [Linear(self.D, self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_in = ModuleList(
            [Linear(self.D, 4 * self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_out = ModuleList(
            [Linear(4 * self.D, self.D, bias=False) for _ in range(self.depth)]
        )
        for layer in list(self.gnn_layers) + list(self.mlp_in) + list(self.mlp_out):
            init.normal_(layer.weight, mean=0.0, std=1.0)
        self.act = ELU()

        # After pooling (if graph-level): linear or 1-hidden-layer decoder; ELU everywhere.
        if self.decoder_type == "linear":
            self.out_proj = Linear(self.D, out_dim, bias=False)
            init.normal_(self.out_proj.weight, mean=0.0, std=sigma_Lp1)
            self.dec_lin1 = None
            self.dec_lin2 = None
            self.dec_lin3 = None
        elif self.decoder_type == "deep":
            self.out_proj = None
            self.dec_lin1 = Linear(self.D, self.D, bias=False)
            self.dec_lin2 = Linear(self.D, out_dim, bias=False)
            init.normal_(self.dec_lin1.weight, mean=0.0, std=sigma_Lp1)
            init.normal_(self.dec_lin2.weight, mean=0.0, std=sigma_Lp1)
            self.dec_lin3 = None
        else:
            self.out_proj = None
            self.dec_lin1 = Linear(self.D, self.D, bias=False)
            self.dec_lin2 = Linear(self.D, self.D, bias=False)
            self.dec_lin3 = Linear(self.D, out_dim, bias=False)
            init.normal_(self.dec_lin1.weight, mean=0.0, std=sigma_Lp1)
            init.normal_(self.dec_lin2.weight, mean=0.0, std=sigma_Lp1)
            init.normal_(self.dec_lin3.weight, mean=0.0, std=sigma_Lp1)

    def get_lr(self, eta_0: float, corr: float = 1.0) -> float:
        if "adam" in self.optimizer:
            return eta_0 / sqrt(self.D)
        return eta_0 * self.D * self.depth

    def get_weight_decay(self, lambda_0: float) -> float:
        return lambda_0 * (self.D) ** (0.5 if "adamw" in self.optimizer else 0)

    def make_param_groups(
        self,
        eta_0: float,
        corr: float = 1.0,
        weight_decay: float = 0.0,
        first_layer_corr: float = 1.0,
    ):
        lr = self.get_lr(eta_0, corr)
        if first_layer_corr != 1.0:
                lr_first = lr * first_layer_corr
        else:
            lr_first = lr

        def split_decay(params):
            decayed, no_decayed = [], []
            for p in params:
                if p.requires_grad:
                    is_bias = p.ndim == 1
                    (no_decayed if is_bias else decayed).append(p)
            groups = []
            if decayed:
                groups.append({"params": decayed, "weight_decay": weight_decay})
            if no_decayed:
                groups.append({"params": no_decayed, "weight_decay": 0.0})
            return groups

        param_groups = []
        embed_params = list(self.embed.parameters())
        if self.decoder_type == "linear":
            last_params = list(self.out_proj.parameters())
        elif self.decoder_type == "deep":
            last_params = list(self.dec_lin1.parameters()) + list(self.dec_lin2.parameters())
        else:
            last_params = (
                list(self.dec_lin1.parameters())
                + list(self.dec_lin2.parameters())
                + list(self.dec_lin3.parameters())
            )
        for name, params, layer_lr in [
            ("first", embed_params, lr_first),
            (
                "resid",
                list(self.gnn_layers.parameters())
                + list(self.mlp_in.parameters())
                + list(self.mlp_out.parameters()),
                lr,
            ),
            ("last", last_params, lr),
        ]:
            for g in split_decay(params):
                g.update({"lr": layer_lr, "name": name})
                param_groups.append(g)
        return param_groups

    def forward(self, data) -> Tensor:
        if self.encoder_type == "lappe":
            if not (hasattr(data, "evals") and hasattr(data, "evecs")):
                raise ValueError(
                    "encoder='lappe' requires data with evals and evecs "
                    "(use a dataset with LapPE=True, e.g. Pascal with --encoder lappe)"
                )
            x = self.embed(data.x, data.evals, data.evecs) / self.sigma_0
        elif self.encoder_type == "deeper":
            x = self.embed(data.x)
        else:
            x = self.embed(data.x) / (sqrt(self.in_dim) * self.sigma_0)
        edge_index, edge_weight = self._get_edge_weight(x, data.edge_index)

        for idx, layer in enumerate(self.gnn_layers):
            if self.layer_norm:
                x = self.ln(x)
            h = layer(self.act(x)) / sqrt(self.D)
            row, col = edge_index[0], edge_index[1]
            msg = h[row] if edge_weight is None else edge_weight.view(-1, 1) * h[row]
            h = scatter(msg, col, dim=0, dim_size=x.size(0), reduce="sum")
            x_tilde = x + (1 / (self.depth * self.gamma)) * h

            if self.layer_norm:
                x_tilde = self.ln(x_tilde)
            mlp_h = self.mlp_in[idx](self.act(x_tilde)) / sqrt(self.D)
            mlp_h = self.act(mlp_h)
            mlp_h = self.mlp_out[idx](mlp_h) / sqrt(4 * self.D)
            x = x_tilde + (1 / self.depth) * mlp_h

        x_nodes = x
        if self.log_batch_stats:
            x_nodes.retain_grad()
            self._last_x = x_nodes

        if self.pool:
            x = global_mean_pool(x_nodes, batch=getattr(data, "batch", None))
        else:
            x = x_nodes

        if self.decoder_type == "linear":
            return self.out_proj(x) / (self.D * self.sigma_Lp1)
        # deep: (1/sqrt(D)) sigma( (1/(sigma_{L+1} sqrt(D))) sigma(X) W1 ) W2
        if self.decoder_type == "deep":
            t = self.act(x)
            z = self.dec_lin1(t) / (self.sigma_Lp1 * sqrt(self.D))
            h = self.act(z)
            return self.dec_lin2(h) / (self.sigma_Lp1 * sqrt(self.D))
        # deeper: add one more hidden layer with same sigma_{L+1}, sqrt(D) scaling.
        t = self.act(x)
        z1 = self.dec_lin1(t) / (self.sigma_Lp1 * sqrt(self.D))
        h1 = self.act(z1)
        z2 = self.dec_lin2(h1) / (self.sigma_Lp1 * sqrt(self.D))
        h2 = self.act(z2)
        return self.dec_lin3(h2) / (self.sigma_Lp1 * sqrt(self.D))

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
