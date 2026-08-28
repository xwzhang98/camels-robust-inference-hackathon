"""The de Santi / CosmoGraphNet MetaLayer GNN and the Jeffrey-Wandelt moment loss.

Spec: arXiv:2302.14101 Eqs. 9-12 (message passing + readout) and Eq. 20 (loss),
cross-checked against PabloVD/CosmoGraphNet ``Source/metalayer.py`` and ``Source/training.py``.

::

    edge:  e_ij' = E([n_i, n_j, e_ij])                       (Eq. 9; global is NOT fed here)
    node:  n_i'  = N([n_i, multipool_j(e_ij'), g])           (Eq. 10)
    read:  y     = F([sum_i n_i, mean_i n_i, max_i n_i, g])  (Eq. 12)

``multipool`` is ``concat[sum, max, mean]`` (Eq. 11). The global attribute ``g = log10(N_g)``
is broadcast into the node model and the readout but never updated (paper footnote 3).
Block 1 uses ``residuals=False`` because its input dims differ from ``hidden``; blocks 2..n
use residuals.

Outputs are ``2 * n_params``: the first half is the posterior mean ``mu``, the second half
``sigma``. ``sigma`` is the raw linear output -- only ``sigma^2`` enters the loss, so its
sign is unconstrained and callers must take ``abs()`` at evaluation time. Use
:meth:`DeSantiGNN.split_output`, which does it for you.

**No ``torch_geometric``.** Upstream uses it for exactly two things, ``utils.scatter`` and
``loader.DataLoader``, and both are replaced below by a dozen lines of torch. PyG is the
hardest install in this stack, and every team that loses a morning to it is a team that did
not do the actual task.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int,
            reduce: str) -> torch.Tensor:
    """Local replacement for ``torch_geometric.utils.scatter`` (``dim=0`` only).

    Matches PyG's convention exactly, including the one that surprises people: for
    ``"max"``, **empty output slots are 0**, because PyG starts from a zeros tensor with
    ``include_self=False``. Occupied slots get the true maximum, which may be negative --
    do NOT clamp it.
    """
    shape = (dim_size,) + tuple(src.shape[1:])
    if reduce == "sum":
        return src.new_zeros(shape).index_add_(0, index, src)
    if reduce == "mean":
        total = src.new_zeros(shape).index_add_(0, index, src)
        count = src.new_zeros(dim_size).index_add_(
            0, index, src.new_ones(index.shape[0]))
        return total / count.clamp(min=1).reshape((-1,) + (1,) * (src.dim() - 1))
    if reduce == "max":
        out = src.new_zeros(shape)
        idx = index.reshape((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
        out.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
        return out
    raise ValueError(f"unsupported reduce: {reduce!r}")


def collate(graphs: list[dict]) -> dict:
    """Batch variable-size graphs into one disjoint graph (replaces PyG's ``DataLoader``).

    Node indices of graph ``k`` are offset by the total node count of graphs ``0..k-1``,
    and ``batch[i]`` says which graph node ``i`` belongs to. That vector is what turns the
    readout's ``scatter`` into a per-catalog pooling.
    """
    xs, eis, eas, us, batches = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        n = int(np.asarray(g["x"]).shape[0])
        xs.append(torch.as_tensor(np.asarray(g["x"])))
        eis.append(torch.as_tensor(np.asarray(g["edge_index"], dtype=np.int64)) + offset)
        eas.append(torch.as_tensor(np.asarray(g["edge_attr"])))
        us.append(torch.as_tensor(np.asarray(g["u"])))
        batches.append(torch.full((n,), i, dtype=torch.long))
        offset += n
    return {
        "x": torch.cat(xs, 0),
        "edge_index": torch.cat(eis, 1),
        "edge_attr": torch.cat(eas, 0),
        "u": torch.stack(us, 0),
        "batch": torch.cat(batches, 0),
    }


def _mlp(sizes: list[int]) -> nn.Sequential:
    """Fully connected + ReLU between layers, no activation on the output."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class EdgeModel(nn.Module):
    """``E([n_i, n_j, e_ij]) -> e_ij'`` (Eq. 9).

    ``node_in == 0`` selects the positions-only variant, where the edge model sees only
    the edge attributes (Eq. 13).
    """

    def __init__(self, node_in: int, edge_in: int, hidden: int, edge_out: int,
                 residuals: bool):
        super().__init__()
        self.node_in = node_in
        self.residuals = residuals
        self.mlp = _mlp([2 * node_in + edge_in, hidden, edge_out])

    def forward(self, src, dst, edge_attr):
        h = edge_attr if self.node_in == 0 else torch.cat([src, dst, edge_attr], dim=-1)
        out = self.mlp(h)
        return out + edge_attr if self.residuals else out


class NodeModel(nn.Module):
    """``N([n_i, sum||max||mean of incident e_ij', g]) -> n_i'`` (Eqs. 10-11)."""

    def __init__(self, node_in: int, edge_out: int, hidden: int, node_out: int,
                 n_global: int, residuals: bool):
        super().__init__()
        self.node_in = node_in
        self.residuals = residuals
        self.mlp = _mlp([node_in + 3 * edge_out + n_global, hidden, node_out])

    def forward(self, x, edge_index, edge_attr, u, batch):
        n = x.shape[0]
        col = edge_index[1]                       # aggregate into the destination node
        parts = [scatter(edge_attr, col, n, "sum"),
                 scatter(edge_attr, col, n, "max"),
                 scatter(edge_attr, col, n, "mean"),
                 u[batch]]
        if self.node_in > 0:
            parts.insert(0, x)
        out = self.mlp(torch.cat(parts, dim=-1))
        return out + x if self.residuals else out


class DeSantiGNN(nn.Module):
    """The full model. ``node_features=0`` reproduces the paper's positions-only variant."""

    def __init__(self, node_features: int = 1, edge_features: int = 3, n_global: int = 1,
                 hidden: int = 64, n_layers: int = 3, n_params: int = 2):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.node_features = node_features
        self.n_params = n_params
        blocks = []
        node_in, edge_in = node_features, edge_features
        for layer in range(n_layers):
            residuals = layer > 0      # block 1 changes dimensionality, so it cannot be residual
            blocks.append(nn.ModuleDict({
                "edge": EdgeModel(node_in, edge_in, hidden, hidden, residuals),
                "node": NodeModel(node_in, hidden, hidden, hidden, n_global, residuals),
            }))
            node_in, edge_in = hidden, hidden
        self.blocks = nn.ModuleList(blocks)
        self.readout = _mlp([3 * hidden + n_global, hidden, hidden, hidden, 2 * n_params])

    def forward(self, x, edge_index, edge_attr, u, batch):
        for blk in self.blocks:
            src, dst = x[edge_index[0]], x[edge_index[1]]
            edge_attr = blk["edge"](src, dst, edge_attr)
            x = blk["node"](x, edge_index, edge_attr, u, batch)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        pooled = torch.cat([
            scatter(x, batch, n_graphs, "sum"),
            scatter(x, batch, n_graphs, "mean"),
            scatter(x, batch, n_graphs, "max"),
            u,
        ], dim=-1)
        return self.readout(pooled)

    @staticmethod
    def split_output(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> ``(mu, sigma)``. ``sigma`` is ``|raw|``: only ``sigma^2`` is constrained."""
        half = y.shape[-1] // 2
        return y[..., :half], y[..., half:].abs()


def moment_loss(y_pred: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Jeffrey & Wandelt moment loss, de Santi Eq. 20.

    ::

        L = log( mean_j sum_i (theta - mu)^2 )
          + log( mean_j sum_i [ (theta - mu)^2 - sigma^2 ]^2 )

    The first term fits the posterior mean; the second makes ``sigma`` an actual
    uncertainty rather than a free parameter. The logs put the two terms on the same scale,
    so the mean term cannot swamp the variance term.

    (The reference implementation averages over the batch where the paper sums; that
    differs by an additive constant and has identical gradients.)
    """
    half = y_pred.shape[-1] // 2
    mu, sigma = y_pred[..., :half], y_pred[..., half:]
    sq_err = (theta - mu) ** 2
    loss_mse = torch.mean(torch.sum(sq_err, dim=-1))
    loss_lfi = torch.mean(torch.sum((sq_err - sigma ** 2) ** 2, dim=-1))
    loss = torch.log(loss_mse) + torch.log(loss_lfi)
    return loss, {"mse": float(loss_mse.item()), "lfi": float(loss_lfi.item())}
