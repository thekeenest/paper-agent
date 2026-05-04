"""
Year-snapshotted heterogeneous graph dataset for collaboration forecasting.

Builds a series of PyG HeteroData snapshots from the KuzuDB knowledge graph,
one per year in the data.  Each snapshot contains:

Node types
----------
  author       — Author nodes (canonical_id → integer index)
  institution  — Institution nodes (ror_id → integer index)

Edge types
----------
  (author, coauthored_with, author)        — from COAUTHORED_WITH edges ≤ year T
  (author, affiliated_at, institution)     — from AFFILIATED_AT edges ≤ year T
  (institution, collaborated_with, institution) — from COLLABORATED_WITH ≤ year T

Node features
-------------
  author:       [industry_flag]  (1 if any affiliation is company, else 0)
  institution:  [org_type_ohe]   one-hot over {education, company, facility, other}

Link prediction target
----------------------
  For horizon H=1, the task is to predict which (institution, institution)
  COLLABORATED_WITH edges appear in year T+1 but NOT in year T.

Usage
-----
::

    from src.v2.analytics.forecasting.dataset import CollabForecastDataset

    ds = CollabForecastDataset(kg_db_path="output/v2/kg")
    snapshots = ds.build_snapshots()   # list of (year, HeteroData, pos_edges, neg_edges)
    train_snaps, test_snaps = ds.split()
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_SEED = int(os.getenv("FORECAST_SEED", "42"))
_DEFAULT_DB = Path(os.getenv("KG_DB_PATH", "output/v2/kg"))

_ORG_TYPE_MAP = {"education": 0, "company": 1, "facility": 2, "government": 3, "nonprofit": 4, "other": 5}
_N_ORG_TYPES = len(_ORG_TYPE_MAP)


@dataclass
class YearSnapshot:
    """One year's heterogeneous graph snapshot."""

    year: int
    data: Any  # torch_geometric.data.HeteroData
    author_ids: list[str]  # index → canonical_id
    inst_ids: list[str]    # index → ror_id
    # link prediction targets for horizon +1
    pos_edges: torch.Tensor = field(default_factory=lambda: torch.zeros(2, 0, dtype=torch.long))
    neg_edges: torch.Tensor = field(default_factory=lambda: torch.zeros(2, 0, dtype=torch.long))


class CollabForecastDataset:
    """Build year-snapshotted heterogeneous graph datasets from the KG.

    Parameters
    ----------
    kg_db_path:
        Path to the KuzuDB database directory.
    min_year:
        Earliest year to include.
    max_year:
        Latest year to include.
    seed:
        Random seed for negative edge sampling.
    neg_sample_ratio:
        Ratio of negative to positive edges for link prediction.
    """

    def __init__(
        self,
        kg_db_path: str | Path | None = None,
        min_year: int = 2019,
        max_year: int = 2023,
        seed: int = _SEED,
        neg_sample_ratio: float = 1.0,
    ) -> None:
        self._db_path = Path(str(kg_db_path or _DEFAULT_DB))
        self._min_year = min_year
        self._max_year = max_year
        self._seed = seed
        self._neg_ratio = neg_sample_ratio
        self._rng = random.Random(seed)
        torch.manual_seed(seed)

    def build_snapshots(self) -> list[YearSnapshot]:
        """Build a snapshot per year and attach next-year link prediction targets.

        Returns
        -------
        list[YearSnapshot]
            Sorted by year.  The last year has empty pos/neg edges (no future).
        """
        from src.v2.kg.schema import open_db, KGSchema

        db, conn = open_db(self._db_path)
        KGSchema.create_all(conn)

        snapshots: list[YearSnapshot] = []
        years = list(range(self._min_year, self._max_year + 1))

        # Build per-year cumulative data
        for year in years:
            snap = self._build_one_snapshot(conn, year)
            snapshots.append(snap)

        # Attach link prediction targets: edges appearing in year+1 but not year
        for i, snap in enumerate(snapshots[:-1]):
            next_snap = snapshots[i + 1]
            snap.pos_edges, snap.neg_edges = self._build_link_targets(
                snap, next_snap
            )

        return snapshots

    def split(
        self,
        train_years: list[int] | None = None,
        test_year: int | None = None,
    ) -> tuple[list[YearSnapshot], list[YearSnapshot]]:
        """Return (train_snapshots, test_snapshots).

        Default: train on all years up to max_year-1, test on max_year.
        """
        snaps = self.build_snapshots()
        if not snaps:
            return [], []
        if test_year is None:
            test_year = snaps[-1].year
        train = [s for s in snaps if s.year < test_year]
        test = [s for s in snaps if s.year == test_year]
        return train, test

    # ── private ───────────────────────────────────────────────────────────────

    def _build_one_snapshot(self, conn: Any, up_to_year: int) -> YearSnapshot:
        """Build a cumulative graph snapshot including all edges ≤ up_to_year."""
        try:
            from torch_geometric.data import HeteroData
        except ImportError as exc:
            raise ImportError("torch-geometric is required for forecasting") from exc

        # Collect nodes
        author_ids = self._fetch_author_ids(conn, up_to_year)
        inst_ids = self._fetch_inst_ids(conn, up_to_year)

        author_idx = {aid: i for i, aid in enumerate(author_ids)}
        inst_idx = {rid: i for i, rid in enumerate(inst_ids)}

        data = HeteroData()

        # Node features: author → industry flag
        n_authors = len(author_ids)
        n_insts = len(inst_ids)

        author_feats = self._author_features(conn, author_ids, inst_idx, up_to_year)
        inst_feats = self._inst_features(conn, inst_ids)

        data["author"].x = author_feats
        data["author"].num_nodes = n_authors
        data["institution"].x = inst_feats
        data["institution"].num_nodes = n_insts

        # Edges: author → author (COAUTHORED_WITH)
        coa_src, coa_dst = self._fetch_coauthor_edges(conn, author_idx, up_to_year)
        if coa_src:
            data["author", "coauthored_with", "author"].edge_index = torch.tensor(
                [coa_src, coa_dst], dtype=torch.long
            )

        # Edges: author → institution (AFFILIATED_AT)
        aff_src, aff_dst = self._fetch_affiliation_edges(conn, author_idx, inst_idx, up_to_year)
        if aff_src:
            data["author", "affiliated_at", "institution"].edge_index = torch.tensor(
                [aff_src, aff_dst], dtype=torch.long
            )

        # Edges: institution → institution (COLLABORATED_WITH)
        col_src, col_dst = self._fetch_collab_edges(conn, inst_idx, up_to_year)
        if col_src:
            data["institution", "collaborated_with", "institution"].edge_index = (
                torch.tensor([col_src, col_dst], dtype=torch.long)
            )

        return YearSnapshot(
            year=up_to_year,
            data=data,
            author_ids=author_ids,
            inst_ids=inst_ids,
        )

    def _fetch_author_ids(self, conn: Any, up_to_year: int) -> list[str]:
        res = conn.execute(
            "MATCH (a:Author)-[r:AUTHORED]->(p:Paper)-[pub:PUBLISHED_AT]->(:Venue) "
            "WHERE pub.year <= $yr RETURN DISTINCT a.canonical_id",
            parameters={"yr": up_to_year},
        )
        ids = []
        while res.has_next():
            ids.append(str(res.get_next()[0]))
        return ids

    def _fetch_inst_ids(self, conn: Any, up_to_year: int) -> list[str]:
        res = conn.execute(
            "MATCH (a:Author)-[aff:AFFILIATED_AT]->(i:Institution) "
            "WHERE aff.year <= $yr RETURN DISTINCT i.ror_id",
            parameters={"yr": up_to_year},
        )
        ids = []
        while res.has_next():
            ids.append(str(res.get_next()[0]))
        return ids

    def _author_features(
        self, conn: Any, author_ids: list[str], inst_idx: dict[str, int], up_to_year: int
    ) -> torch.Tensor:
        """Feature vector for each author: [is_industry]."""
        industry_authors: set[str] = set()
        try:
            res = conn.execute(
                "MATCH (a:Author)-[aff:AFFILIATED_AT]->(i:Institution) "
                "WHERE aff.year <= $yr AND i.org_type = 'company' "
                "RETURN DISTINCT a.canonical_id",
                parameters={"yr": up_to_year},
            )
            while res.has_next():
                industry_authors.add(str(res.get_next()[0]))
        except Exception:
            pass
        feats = torch.zeros(len(author_ids), 1, dtype=torch.float)
        for i, aid in enumerate(author_ids):
            if aid in industry_authors:
                feats[i, 0] = 1.0
        return feats

    def _inst_features(self, conn: Any, inst_ids: list[str]) -> torch.Tensor:
        """Feature vector for each institution: one-hot org_type."""
        org_types: dict[str, str] = {}
        try:
            res = conn.execute("MATCH (i:Institution) RETURN i.ror_id, i.org_type")
            while res.has_next():
                row = res.get_next()
                org_types[str(row[0])] = str(row[1])
        except Exception:
            pass
        feats = torch.zeros(len(inst_ids), _N_ORG_TYPES, dtype=torch.float)
        for i, rid in enumerate(inst_ids):
            ot = org_types.get(rid, "other")
            idx = _ORG_TYPE_MAP.get(ot, _ORG_TYPE_MAP["other"])
            feats[i, idx] = 1.0
        return feats

    def _fetch_coauthor_edges(
        self, conn: Any, author_idx: dict[str, int], up_to_year: int
    ) -> tuple[list[int], list[int]]:
        src, dst = [], []
        try:
            res = conn.execute(
                "MATCH (a:Author)-[e:COAUTHORED_WITH]->(b:Author) "
                "WHERE e.year <= $yr RETURN a.canonical_id, b.canonical_id",
                parameters={"yr": up_to_year},
            )
            while res.has_next():
                row = res.get_next()
                ai, bi = author_idx.get(str(row[0])), author_idx.get(str(row[1]))
                if ai is not None and bi is not None:
                    src.append(ai)
                    dst.append(bi)
        except Exception:
            pass
        return src, dst

    def _fetch_affiliation_edges(
        self,
        conn: Any,
        author_idx: dict[str, int],
        inst_idx: dict[str, int],
        up_to_year: int,
    ) -> tuple[list[int], list[int]]:
        src, dst = [], []
        try:
            res = conn.execute(
                "MATCH (a:Author)-[aff:AFFILIATED_AT]->(i:Institution) "
                "WHERE aff.year <= $yr RETURN DISTINCT a.canonical_id, i.ror_id",
                parameters={"yr": up_to_year},
            )
            while res.has_next():
                row = res.get_next()
                ai, ii = author_idx.get(str(row[0])), inst_idx.get(str(row[1]))
                if ai is not None and ii is not None:
                    src.append(ai)
                    dst.append(ii)
        except Exception:
            pass
        return src, dst

    def _fetch_collab_edges(
        self, conn: Any, inst_idx: dict[str, int], up_to_year: int
    ) -> tuple[list[int], list[int]]:
        src, dst = [], []
        try:
            res = conn.execute(
                "MATCH (i:Institution)-[c:COLLABORATED_WITH]->(j:Institution) "
                "WHERE c.year <= $yr RETURN DISTINCT i.ror_id, j.ror_id",
                parameters={"yr": up_to_year},
            )
            while res.has_next():
                row = res.get_next()
                ii, ji = inst_idx.get(str(row[0])), inst_idx.get(str(row[1]))
                if ii is not None and ji is not None:
                    src.append(ii)
                    dst.append(ji)
        except Exception:
            pass
        return src, dst

    def _build_link_targets(
        self, snap: YearSnapshot, next_snap: YearSnapshot
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute positive edges (new in next year) and sample negatives."""
        inst_idx = {rid: i for i, rid in enumerate(snap.inst_ids)}
        next_inst_idx = {rid: i for i, rid in enumerate(next_snap.inst_ids)}

        # Edges present in current snapshot
        cur_edges: set[tuple[int, int]] = set()
        ei_key = ("institution", "collaborated_with", "institution")
        if ei_key in snap.data.edge_types:
            ei = snap.data[ei_key].edge_index
            for k in range(ei.shape[1]):
                cur_edges.add((int(ei[0, k]), int(ei[1, k])))

        # Edges present in next snapshot — re-indexed to current inst_idx
        new_edges: list[tuple[int, int]] = []
        if ei_key in next_snap.data.edge_types:
            ei_next = next_snap.data[ei_key].edge_index
            for k in range(ei_next.shape[1]):
                src_id = next_snap.inst_ids[int(ei_next[0, k])]
                dst_id = next_snap.inst_ids[int(ei_next[1, k])]
                si = inst_idx.get(src_id)
                di = inst_idx.get(dst_id)
                if si is not None and di is not None and (si, di) not in cur_edges:
                    new_edges.append((si, di))

        if not new_edges:
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(2, 0, dtype=torch.long),
            )

        pos = torch.tensor(new_edges, dtype=torch.long).T  # [2, E]

        # Negative sampling: random (i, j) not in current or next edges
        all_edges = cur_edges | set(new_edges)
        n_inst = len(snap.inst_ids)
        n_neg = int(len(new_edges) * self._neg_ratio)
        negs: list[tuple[int, int]] = []
        attempts = 0
        while len(negs) < n_neg and attempts < n_neg * 20:
            i = self._rng.randint(0, n_inst - 1)
            j = self._rng.randint(0, n_inst - 1)
            if i != j and (i, j) not in all_edges:
                negs.append((i, j))
            attempts += 1

        neg = torch.tensor(negs, dtype=torch.long).T if negs else torch.zeros(2, 0, dtype=torch.long)
        return pos, neg
