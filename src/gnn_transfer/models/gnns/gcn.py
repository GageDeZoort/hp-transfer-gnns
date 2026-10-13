from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor
from torch.nn import ELU, Dropout, Module, init
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_remaining_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


def gcn_norm(
    edge_index: Tensor,
    edge_weight: Tensor = None,
    num_nodes: int = None,
    res_agg: float = 1.0,
):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1)).to(edge_index.device)

    edge_index, tmp_edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, 1.0, num_nodes
    )
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

    # form the matrix P = D^{-1/2} @ A @ D^{-1/2}
    row, col = edge_index[0], edge_index[1]
    not_self_loops = torch.ne(row, col).bool()
    deg = scatter(
        edge_weight[not_self_loops],  # no self-loops
        col[not_self_loops],
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    )
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    weights = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # add self-loops as residual aggregation, (1-t)*I + t*P
    self_loops = torch.eq(row, col).long()
    weighted_prop = (
        1 - res_agg
    ) * self_loops + res_agg * weights * not_self_loops.long()
    return edge_index, weighted_prop


ADJACENCY_TYPES = ("normalized", "unnormalized")


def build_mp_operator(
    edge_index: Tensor,
    num_nodes: int,
    adjacency: str,
    res_agg: float,
    dtype: torch.dtype,
):
    """Build the message-passing operator P for a residual GNN stack.

    Returns ``(edge_index, edge_weight)`` for the selected ``adjacency`` strategy:
    ``"normalized"`` gives the degree-normalized operator from :func:`gcn_norm`,
    ``"unnormalized"`` gives raw adjacency weights with the same
    ``(1 - res_agg) * I + res_agg * A`` residual-aggregation convention.
    """
    if adjacency not in ADJACENCY_TYPES:
        raise ValueError(f"adjacency must be one of {ADJACENCY_TYPES}, got {adjacency!r}")

    if adjacency == "normalized":
        return gcn_norm(edge_index, None, num_nodes, res_agg=res_agg)

    edge_index, _ = add_remaining_self_loops(edge_index, None, 1.0, num_nodes)
    row, col = edge_index[0], edge_index[1]
    self_loops = torch.eq(row, col).to(dtype=dtype)
    not_self_loops = torch.ne(row, col).to(dtype=dtype)
    edge_weight = (1 - res_agg) * self_loops + res_agg * not_self_loops
    return edge_index, edge_weight


class GCN(MessagePassing):
    _cached_edge_index: tuple[Tensor, Tensor] | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cached: bool,
        res_agg: float,
        activation: Module = ELU,
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            in_dim: input vertex dimension
            out_dim: output vertex dimension
            cached: cache edge quantities for single-graph learning tasks
            activation: non-linearity (provide as Torch module)
        """
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.res_agg = res_agg

        # cache edges for single-graph learning tasks
        self.cached = cached
        self._cached_edge_index = None

        self.dropout = Dropout(p=dropout)
        self.sigma = activation()

        self.W = Linear(self.in_dim, self.out_dim, bias=False)
        self.reset_parameters(self.W)

    def reset_parameters(self, weights):
        weights.reset_parameters()
        self._cached_edge_index = None
        for p in weights.parameters():
            init.normal_(p.data, mean=0, std=1)

    def get_edge_weight(self, x, edge_index, edge_weight=None):
        cache = self._cached_edge_index
        if cache is None:
            if edge_weight is not None and self.cached:
                self._cached_edge_index = (edge_index, edge_weight)
            else:
                if self._cached_edge_index is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        res_agg=self.res_agg,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
            return edge_index, edge_weight
        else:
            return cache[0], cache[1]

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        deg_inv: OptTensor = None,
        **kwargs,
    ) -> Tensor:

        if deg_inv is not None:
            edge_weight = deg_inv
        else:
            edge_index, edge_weight = self.get_edge_weight(x, edge_index, deg_inv)
        out = self.propagate(
            edge_index,
            x=self.W(x) / sqrt(self.in_dim),
            edge_weight=edge_weight,
            size=None,
        )
        return out, None

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
