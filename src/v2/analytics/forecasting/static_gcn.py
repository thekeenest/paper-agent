"""
Baseline: Static GCN.

A single-snapshot homogeneous GCN trained on the final cumulative
institution collaboration graph.  Unlike DHGNNLite, it ignores temporal
dynamics and processes all years as one static graph.

Architecture
------------
  2-layer GCN on institution nodes (features: org_type one-hot)
  → link decoder: inner product + sigmoid

Usage
-----
::

    from src.v2.analytics.forecasting.static_gcn import StaticGCN, train_static_gcn

    model = StaticGCN(in_channels=6, hidden=64)
    train_static_gcn(model, train_snapshots, epochs=100, lr=1e-3)
    auc, ap = evaluate_static_gcn(model, test_snapshots)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_SEED = int(os.getenv("FORECAST_SEED", "42"))


class GCNConv(nn.Module):
    """Simple GCN convolution (mean aggregation, no edge weights)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        # x: [N, F]; edge_index: [2, E]
        out = self.lin(x)
        if edge_index.shape[1] == 0:
            return out
        # Mean aggregate
        src = edge_index[0]
        dst = edge_index[1]
        agg = torch.zeros_like(out)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, out.shape[1]), out[src])
        # Normalise by degree
        deg = torch.zeros(n_nodes, dtype=torch.float)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0]))
        deg = deg.clamp(min=1).unsqueeze(-1)
        return agg / deg + out


class StaticGCN(nn.Module):
    """Two-layer GCN for homogeneous institution graph.

    Parameters
    ----------
    in_channels:
        Institution feature dimensionality (default 6 for org_type one-hot).
    hidden:
        Hidden dimensionality.
    """

    def __init__(self, in_channels: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.link_decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        n = x.shape[0]
        h = F.elu(self.conv1(x, edge_index, n))
        h = self.conv2(h, edge_index, n)
        return h

    def decode_link(
        self, h: torch.Tensor, edge_pairs: torch.Tensor
    ) -> torch.Tensor:
        if edge_pairs.shape[1] == 0:
            return torch.zeros(0)
        h_src = h[edge_pairs[0]]
        h_dst = h[edge_pairs[1]]
        logits = self.link_decoder(torch.cat([h_src, h_dst], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_pairs: torch.Tensor
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.decode_link(h, edge_pairs)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────── training ────────────────────────────────────────


def _build_static_graph(snapshots: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge all snapshots into one static institution graph."""
    if not snapshots:
        return torch.zeros(0, 6), torch.zeros(2, 0, dtype=torch.long)

    # Use the last snapshot's features (most complete)
    last = snapshots[-1]
    x = last.data["institution"].x if hasattr(last.data["institution"], "x") else torch.zeros(0, 6)

    ei_key = ("institution", "collaborated_with", "institution")
    edges: set[tuple[int, int]] = set()
    for snap in snapshots:
        if ei_key in snap.data.edge_types:
            ei = snap.data[ei_key].edge_index
            for k in range(ei.shape[1]):
                edges.add((int(ei[0, k]), int(ei[1, k])))

    if not edges:
        return x, torch.zeros(2, 0, dtype=torch.long)

    edge_index = torch.tensor(list(edges), dtype=torch.long).T
    return x.float(), edge_index


def train_static_gcn(
    model: StaticGCN,
    train_snapshots: list[Any],
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = _SEED,
    verbose: bool = True,
) -> list[float]:
    """Train StaticGCN on accumulated training snapshots.

    Returns
    -------
    list[float]
        Per-epoch training losses.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x, edge_index = _build_static_graph(train_snapshots)

    target_snaps = [s for s in train_snapshots if s.pos_edges.shape[1] > 0]
    if not target_snaps or x.shape[0] == 0:
        return []

    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        epoch_loss = 0.0
        for snap in target_snaps:
            pos = snap.pos_edges
            neg = snap.neg_edges
            if pos.shape[1] == 0:
                continue
            all_edges = torch.cat([pos, neg], dim=1)
            labels = torch.cat(
                [torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]
            )
            preds = model(x, edge_index, all_edges)
            if preds.shape[0] != labels.shape[0]:
                continue
            loss = F.binary_cross_entropy(preds, labels)
            loss.backward()
            epoch_loss += loss.item()

        optimizer.step()
        losses.append(epoch_loss / max(len(target_snaps), 1))

        if verbose and epoch % 20 == 0:
            print(f"StaticGCN epoch {epoch:4d}/{epochs}  loss={losses[-1]:.4f}")

    return losses


def evaluate_static_gcn(
    model: StaticGCN,
    test_snapshots: list[Any],
    train_snapshots: list[Any] | None = None,
) -> tuple[float, float]:
    """Evaluate link prediction AUC and AP."""
    from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import]

    all_snaps = (train_snapshots or []) + test_snapshots
    x, edge_index = _build_static_graph(all_snaps)

    if x.shape[0] == 0:
        return 0.5, 0.0

    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for snap in test_snapshots:
            if snap.pos_edges.shape[1] == 0:
                continue
            pos = snap.pos_edges
            neg = snap.neg_edges
            all_edges = torch.cat([pos, neg], dim=1)
            labels = torch.cat(
                [torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]
            )
            preds = model(x, edge_index, all_edges)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    if len(set(all_labels)) < 2:
        return 0.5, 0.0

    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    return float(auc), float(ap)
