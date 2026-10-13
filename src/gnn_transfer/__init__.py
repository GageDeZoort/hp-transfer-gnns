"""Hyperparameter transfer for graph neural networks.

Public companion package for *Hyperparameter Transfer in Graph Neural Networks*
(LoG Conference proceedings track).
"""

from gnn_transfer.models.gnns.simple_transfer_gnn import SimpleGNN
from gnn_transfer.models.gnns.standard_gnn import StandardGCN

__all__ = ["SimpleGNN", "StandardGCN"]
__version__ = "0.1.0"
