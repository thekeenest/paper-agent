"""
Baseline: Co-occurrence Frequency (co_freq).

Predicts future collaboration probability based on the co-occurrence
frequency of institutions up to time T.

Score(i, j, T) = papers_count(i, j, T) / max_papers_count(T)
(normalised to [0, 1])

This is the simplest non-trivial baseline — it captures the intuition
that frequently collaborating institutions are likely to collaborate again.

Usage
-----
::

    from src.v2.analytics.forecasting.co_freq import CoFreqBaseline

    model = CoFreqBaseline()
    auc, ap = model.evaluate(test_snapshots, train_snapshots)
"""
from __future__ import annotations

from typing import Any

import torch


class CoFreqBaseline:
    """Co-occurrence frequency baseline for institution link prediction.

    No learned parameters.  Scores are derived from the
    COLLABORATED_WITH edge ``papers_count`` attribute.
    """

    def predict(
        self,
        snapshot: Any,  # YearSnapshot
        edge_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Return co-freq scores for each (src, dst) pair.

        Parameters
        ----------
        snapshot:
            YearSnapshot up to time T.
        edge_pairs:
            [2, E] institution index pairs to score.

        Returns
        -------
        torch.Tensor  [E]  scores in [0, 1].
        """
        # Build papers_count matrix
        data = snapshot.data
        ei_key = ("institution", "collaborated_with", "institution")

        # Default: score 0 (no history)
        n_inst = data["institution"].num_nodes
        freq: dict[tuple[int, int], float] = {}

        if ei_key in data.edge_types:
            ei = data[ei_key].edge_index
            # papers_count is not stored on edges in HeteroData by default;
            # treat each edge as count=1 (binary presence)
            for k in range(ei.shape[1]):
                src = int(ei[0, k])
                dst = int(ei[1, k])
                freq[(src, dst)] = freq.get((src, dst), 0) + 1

        if not freq:
            return torch.zeros(edge_pairs.shape[1])

        max_count = max(freq.values()) if freq else 1.0
        scores = torch.tensor(
            [freq.get((int(edge_pairs[0, k]), int(edge_pairs[1, k])), 0.0) / max_count
             for k in range(edge_pairs.shape[1])],
            dtype=torch.float,
        )
        return scores

    def evaluate(
        self,
        test_snapshots: list[Any],
        train_snapshots: list[Any] | None = None,
    ) -> tuple[float, float]:
        """Compute AUC and AP for the test snapshots.

        Returns
        -------
        tuple[float, float]
            (AUC, AP)
        """
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import]

        all_preds: list[float] = []
        all_labels: list[float] = []

        for snap in test_snapshots:
            if snap.pos_edges.shape[1] == 0:
                continue

            pos = snap.pos_edges
            neg = snap.neg_edges
            all_edges = torch.cat([pos, neg], dim=1)
            labels = torch.cat(
                [torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]
            )

            preds = self.predict(snap, all_edges)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        if len(set(all_labels)) < 2:
            return 0.5, 0.0

        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        return float(auc), float(ap)
