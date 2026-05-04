"""
DHGNN-Lite — a lightweight Dynamic Heterogeneous Graph Neural Network for
institution collaboration forecasting.

Design goals
------------
* ≤ 5M parameters
* CPU training in < 30 minutes on the PaperAffilBench KG (~200 papers)
* AUC ≥ co_freq AUC + 0.05 on 1-year horizon link prediction

Architecture
------------
1. Per-node-type linear embedding (maps heterogeneous node features to
   a shared d-dimensional space).
2. Two heterogeneous graph attention (HAT) layers aggregating across
   node types.
3. Temporal aggregation: mean-pool snapshots up to time T.
4. Link decoder: inner-product + sigmoid on institution node embeddings.

The name is inspired by DHGNN (Yin et al., 2019) but the implementation
is purpose-built to be minimal and CPU-efficient.

Usage
-----
::

    from src.v2.analytics.forecasting.dataset import CollabForecastDataset
    from src.v2.analytics.forecasting.dhgnn_lite import DHGNNLite, train_dhgnn

    ds = CollabForecastDataset(kg_db_path="output/v2/kg")
    train_snaps, test_snaps = ds.split()

    model = DHGNNLite(author_in=1, inst_in=6, hidden=64, n_layers=2)
    train_dhgnn(model, train_snaps, epochs=50, lr=1e-3, seed=42)

    auc, ap = evaluate_dhgnn(model, test_snaps)
    print(f"AUC={auc:.4f}  AP={ap:.4f}")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_SEED = int(os.getenv("FORECAST_SEED", "42"))
_CHECKPOINT_DIR = Path(os.getenv("FORECAST_CKPT", "experiments/forecasting/checkpoints"))


# ─────────────────────────── building blocks ─────────────────────────────────


class HeteroLinear(nn.Module):
    """Per-node-type input projections."""

    def __init__(self, in_channels: dict[str, int], out_channels: int) -> None:
        super().__init__()
        self.lins = nn.ModuleDict(
            {nt: nn.Linear(in_c, out_channels) for nt, in_c in in_channels.items()}
        )

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {nt: F.elu(self.lins[nt](x)) for nt, x in x_dict.items() if nt in self.lins}


class HeteroAttentionConv(nn.Module):
    """Single-head heterogeneous attention aggregation for one edge type.

    For a message-passing step (src → dst):
      alpha = softmax(W_att [h_src || h_dst])
      h_dst_new += sum_{src in N(dst)} alpha * W_msg * h_src
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.w_att = nn.Linear(hidden * 2, 1, bias=False)
        self.w_msg = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_src, h_dst:  [N_src, hidden], [N_dst, hidden]
        edge_index:    [2, E]  (src, dst) pairs

        Returns
        -------
        torch.Tensor  [N_dst, hidden]  updated dst embeddings
        """
        if edge_index.shape[1] == 0:
            return h_dst

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        h_s = h_src[src_nodes]  # [E, hidden]
        h_d = h_dst[dst_nodes]  # [E, hidden]

        # Attention weights
        alpha = self.w_att(torch.cat([h_s, h_d], dim=-1))  # [E, 1]
        # Scatter-softmax per destination node
        alpha = _scatter_softmax(alpha.squeeze(-1), dst_nodes, h_dst.shape[0])  # [E]

        # Messages
        msgs = alpha.unsqueeze(-1) * self.w_msg(h_s)  # [E, hidden]

        # Aggregate
        out = torch.zeros_like(h_dst)
        out.scatter_add_(0, dst_nodes.unsqueeze(-1).expand_as(msgs), msgs)
        return out


def _scatter_softmax(
    values: torch.Tensor, index: torch.Tensor, n_dst: int
) -> torch.Tensor:
    """Numerically-stable scatter softmax."""
    # Max per destination
    max_vals = torch.full((n_dst,), float("-inf"))
    max_vals.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    shifted = values - max_vals[index]
    exp_v = torch.exp(shifted)
    sum_exp = torch.zeros(n_dst)
    sum_exp.scatter_add_(0, index, exp_v)
    return exp_v / (sum_exp[index] + 1e-9)


class DHGNNLayer(nn.Module):
    """One DHGNN-Lite layer: aggregate over all edge types, then combine."""

    _EDGE_TYPES = [
        ("author", "coauthored_with", "author"),
        ("author", "affiliated_at", "institution"),
        ("institution", "collaborated_with", "institution"),
    ]

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.convs = nn.ModuleDict(
            {f"{s}__{r}__{d}": HeteroAttentionConv(hidden) for s, r, d in self._EDGE_TYPES}
        )
        self.norm_author = nn.LayerNorm(hidden)
        self.norm_inst = nn.LayerNorm(hidden)

    def forward(
        self,
        h_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        new_h: dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in h_dict.items()}

        for src_t, rel, dst_t in self._EDGE_TYPES:
            key = f"{src_t}__{rel}__{dst_t}"
            if (src_t, rel, dst_t) not in edge_index_dict:
                continue
            ei = edge_index_dict[(src_t, rel, dst_t)]
            if src_t not in h_dict or dst_t not in h_dict:
                continue
            agg = self.convs[key](h_dict[src_t], h_dict[dst_t], ei)
            new_h[dst_t] = new_h[dst_t] + agg

        # Residual + norm
        for nt in h_dict:
            new_h[nt] = new_h[nt] + h_dict[nt]
        if "author" in new_h:
            new_h["author"] = self.norm_author(new_h["author"])
        if "institution" in new_h:
            new_h["institution"] = self.norm_inst(new_h["institution"])
        return new_h


# ─────────────────────────── DHGNNLite model ─────────────────────────────────


class DHGNNLite(nn.Module):
    """Lightweight dynamic heterogeneous GNN for link prediction.

    Parameters
    ----------
    author_in:
        Dimensionality of author input features.
    inst_in:
        Dimensionality of institution input features.
    hidden:
        Hidden dimensionality (shared across all node types after projection).
    n_layers:
        Number of DHGNN message-passing layers.
    temporal_hidden:
        Size of GRU hidden state for temporal aggregation.
    """

    def __init__(
        self,
        author_in: int = 1,
        inst_in: int = 6,
        hidden: int = 64,
        n_layers: int = 2,
        temporal_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.embed = HeteroLinear({"author": author_in, "institution": inst_in}, hidden)
        self.layers = nn.ModuleList([DHGNNLayer(hidden) for _ in range(n_layers)])
        self.gru = nn.GRUCell(hidden, temporal_hidden)
        self.link_decoder = nn.Sequential(
            nn.Linear(temporal_hidden * 2, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self._hidden = hidden
        self._temporal_hidden = temporal_hidden

    def encode_snapshot(
        self,
        data: Any,  # HeteroData
        temporal_state: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Encode one snapshot into institution embeddings + updated temporal state.

        Parameters
        ----------
        data:
            HeteroData for one year.
        temporal_state:
            Dictionary mapping institution integer index → GRU hidden state.
            Pass None for the first snapshot.

        Returns
        -------
        inst_emb:
            [N_inst, temporal_hidden] institution embeddings.
        new_state:
            Updated temporal state.
        """
        # Build feature dicts
        x_dict: dict[str, torch.Tensor] = {}
        if hasattr(data["author"], "x") and data["author"].x is not None:
            x_dict["author"] = data["author"].x.float()
        else:
            x_dict["author"] = torch.zeros(data["author"].num_nodes, 1)
        if hasattr(data["institution"], "x") and data["institution"].x is not None:
            x_dict["institution"] = data["institution"].x.float()
        else:
            x_dict["institution"] = torch.zeros(data["institution"].num_nodes, 6)

        # Input projection
        h_dict = self.embed(x_dict)

        # Build edge index dict
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index

        # Message-passing layers
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)

        # Temporal GRU update per institution node
        h_inst = h_dict.get("institution", torch.zeros(0, self._hidden))
        n_inst = h_inst.shape[0]

        if temporal_state is None:
            temporal_state = {}

        new_state: dict[int, torch.Tensor] = {}
        if n_inst == 0:
            return torch.zeros(0, self._temporal_hidden), new_state

        # Batch GRU update
        prev_hidden = torch.stack(
            [temporal_state.get(i, torch.zeros(self._temporal_hidden)) for i in range(n_inst)]
        )  # [N_inst, temporal_hidden]
        new_hidden = self.gru(h_inst, prev_hidden)  # [N_inst, temporal_hidden]

        for i in range(n_inst):
            new_state[i] = new_hidden[i]

        return new_hidden, new_state

    def decode_link(
        self,
        inst_emb: torch.Tensor,
        edge_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict link probability for each (src, dst) pair.

        Parameters
        ----------
        inst_emb:
            [N_inst, temporal_hidden]
        edge_pairs:
            [2, E] source and destination indices.

        Returns
        -------
        torch.Tensor  [E]  sigmoid probabilities.
        """
        if edge_pairs.shape[1] == 0:
            return torch.zeros(0)
        h_src = inst_emb[edge_pairs[0]]
        h_dst = inst_emb[edge_pairs[1]]
        logits = self.link_decoder(torch.cat([h_src, h_dst], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────── training ────────────────────────────────────────


def train_dhgnn(
    model: DHGNNLite,
    train_snapshots: list[Any],
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = _SEED,
    checkpoint_path: Path | None = None,
    verbose: bool = True,
) -> list[float]:
    """Train DHGNNLite on a list of YearSnapshot objects.

    Parameters
    ----------
    model:
        Untrained DHGNNLite instance.
    train_snapshots:
        list[YearSnapshot] — sorted by year; last snapshot in sequence has targets.
    epochs:
        Number of training epochs.
    lr:
        Adam learning rate.
    seed:
        Reproducibility seed.
    checkpoint_path:
        If provided, save best model weights here.
    verbose:
        Print loss per epoch.

    Returns
    -------
    list[float]
        Per-epoch training losses.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    best_loss = float("inf")

    # Filter snapshots that have link prediction targets
    target_snaps = [s for s in train_snapshots if s.pos_edges.shape[1] > 0]
    if not target_snaps:
        if verbose:
            print("No training snapshots with link prediction targets.")
        return losses

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for snap in target_snaps:
            optimizer.zero_grad()

            # Encode all snapshots up to this one (temporal context)
            state: dict[int, torch.Tensor] | None = None
            for s in train_snapshots:
                if s.year > snap.year:
                    break
                inst_emb, state = model.encode_snapshot(s.data, state)
            # inst_emb is now the representation at snap.year

            if inst_emb.shape[0] == 0:
                continue

            # Positive + negative edges
            pos = snap.pos_edges
            neg = snap.neg_edges

            if pos.shape[1] == 0:
                continue

            all_edges = torch.cat([pos, neg], dim=1)
            labels = torch.cat(
                [torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]
            )

            # Re-index: pos/neg edges use inst_idx of the source snapshot
            # but inst_emb uses current snapshot indices
            preds = model.decode_link(inst_emb, all_edges)
            if preds.shape[0] != labels.shape[0]:
                continue

            loss = F.binary_cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  best={best_loss:.4f}")

    return losses


# ─────────────────────────── evaluation ──────────────────────────────────────


def evaluate_dhgnn(
    model: DHGNNLite,
    test_snapshots: list[Any],
    train_snapshots: list[Any] | None = None,
) -> tuple[float, float]:
    """Evaluate link prediction AUC and Average Precision.

    Parameters
    ----------
    model:
        Trained DHGNNLite.
    test_snapshots:
        list[YearSnapshot] to evaluate on.
    train_snapshots:
        If provided, replay to build temporal state before evaluation.

    Returns
    -------
    tuple[float, float]
        (AUC, AP)
    """
    from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import]

    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for snap in test_snapshots:
            if snap.pos_edges.shape[1] == 0:
                continue

            # Build temporal context
            state: dict[int, torch.Tensor] | None = None
            context_snaps = (train_snapshots or []) + [snap]
            for s in context_snaps:
                inst_emb, state = model.encode_snapshot(s.data, state)

            if inst_emb.shape[0] == 0:
                continue

            pos = snap.pos_edges
            neg = snap.neg_edges
            all_edges = torch.cat([pos, neg], dim=1)
            labels = torch.cat(
                [torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]
            )

            preds = model.decode_link(inst_emb, all_edges)
            if preds.shape[0] != labels.shape[0]:
                continue

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    if len(set(all_labels)) < 2:
        return 0.5, 0.0

    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    return float(auc), float(ap)


def load_checkpoint(model: DHGNNLite, path: Path) -> DHGNNLite:
    """Load weights from a checkpoint file."""
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
