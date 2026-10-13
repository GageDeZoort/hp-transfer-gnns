# Hyperparameter Transfer in GNNs

Public companion code for **Hyperparameter Transfer in Graph Neural Networks**
(Learning on Graphs Conference, proceedings track).

This repo ships a minimal residual GCN under a **transfer parameterization**
(`SimpleGNN`) and an architecturally matched **standard parameterization**
baseline (`StandardGCN`), plus a Colab tutorial that shows learning-rate transfer
on CiteSeer.

## Why this matters

Tuning learning rates at large width/depth is expensive. A good parameterization
lets you tune a cheap proxy model and reuse the same base learning rate $\eta_0$
at scale. For Adam on residual GNNs, the recipe is essentially:

| Piece | Transfer rule |
| --- | --- |
| Residual updates | $1/L$ branch scaling |
| Adam LR | $\eta = \eta_0 / \sqrt{D}$ |
| Encoder / decoder init | $\sigma_0 = \sigma_{L+1} = 1/\sqrt{D}$ |

Under standard (fan-in) parameterization the same $\eta_0$ does **not** transfer
across width — you have to retune.

## Quickstart (Colab)

Open [`notebooks/hyperparameter_transfer_citeseer.ipynb`](notebooks/hyperparameter_transfer_citeseer.ipynb)
with a GPU runtime. The notebook installs dependencies, loads CiteSeer, sweeps
$\eta_0$ at a proxy width, then trains a wider model at the transferred optimum.

## Local install

```bash
pip install -e .
```

```python
from gnn_transfer import SimpleGNN, StandardGCN
```

## Relation to the research codebase

The research experiments live in a private/full repo (`hp-transfer-gts`).
This package keeps only the public model surface. See
[`COPY_MANIFEST.md`](COPY_MANIFEST.md) for the exact file list.

## Citation

If you use this code, please cite the LoG paper:

```bibtex
@inproceedings{dezoort2025hptransfergnn,
  title     = {Hyperparameter Transfer in Graph Neural Networks},
  author    = {Anonymous},
  booktitle = {Learning on Graphs Conference (Proceedings Track)},
  year      = {2025}
}
```

Update the BibTeX once the camera-ready authorship is public.
