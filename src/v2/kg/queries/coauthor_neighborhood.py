"""
Query: coauthor_neighborhood
-----------------------------
Return the 1- or 2-hop co-author graph centred on a given author
(identified by canonical_id or name substring).

Return type
-----------
CoauthorNeighborhoodResult — contains a node list and an edge list,
suitable for downstream visualisation or GraphRAG context injection.

Cypher strategy
---------------
1-hop: direct COAUTHORED_WITH neighbours.
2-hop: extend through each neighbour's neighbours, keeping depth ≤ 2.

Usage
-----
::

    from src.v2.kg.queries.coauthor_neighborhood import coauthor_neighborhood

    result = coauthor_neighborhood(conn, author_id="s2:12345", hops=1)
    print(result.center_id, len(result.nodes), len(result.edges))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoauthorNode:
    """A single author node in the neighbourhood subgraph."""

    canonical_id: str
    name_variants: list[str]
    hop: int  # 0 = center, 1 = direct co-author, 2 = 2nd-order


@dataclass
class CoauthorEdge:
    """A COAUTHORED_WITH edge in the neighbourhood subgraph."""

    src: str
    dst: str
    paper_id: str
    year: int


@dataclass
class CoauthorNeighborhoodResult:
    """Container returned by :func:`coauthor_neighborhood`."""

    center_id: str
    nodes: list[CoauthorNode] = field(default_factory=list)
    edges: list[CoauthorEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_id": self.center_id,
            "nodes": [
                {
                    "canonical_id": n.canonical_id,
                    "name_variants": n.name_variants,
                    "hop": n.hop,
                }
                for n in self.nodes
            ],
            "edges": [
                {"src": e.src, "dst": e.dst, "paper_id": e.paper_id, "year": e.year}
                for e in self.edges
            ],
        }


def coauthor_neighborhood(
    conn: Any,
    author_id: str | None = None,
    author_name: str | None = None,
    hops: int = 1,
    max_nodes: int = 50,
) -> CoauthorNeighborhoodResult:
    """Return the co-author neighbourhood of an author.

    Parameters
    ----------
    conn:
        Open ``kuzu.Connection``.
    author_id:
        Exact ``canonical_id`` of the center author.  Takes priority over
        ``author_name``.
    author_name:
        Substring match against ``name_variants`` (case-insensitive).
        Used only if ``author_id`` is ``None``.
    hops:
        Neighbourhood depth: 1 or 2.
    max_nodes:
        Hard cap on returned nodes (excluding center) to avoid huge subgraphs.

    Returns
    -------
    CoauthorNeighborhoodResult
    """
    if hops not in (1, 2):
        raise ValueError(f"hops must be 1 or 2, got {hops}")

    # ── resolve center author ────────────────────────────────────────────────
    if author_id:
        center_id = author_id
    elif author_name:
        # KuzuDB 0.11.3 has a bug with any() + parameters; work around by
        # searching the canonical_id (which embeds the normalised name) and
        # falling back to a Python-side full-scan for name_variants.
        name_key = author_name.replace(" ", "_").lower()
        res = conn.execute(
            "MATCH (a:Author) WHERE LOWER(a.canonical_id) CONTAINS $nk "
            "RETURN a.canonical_id LIMIT 1",
            parameters={"nk": name_key},
        )
        if res.has_next():
            center_id = str(res.get_next()[0])
        else:
            # Full-scan fallback with Python-side name_variants matching
            res_all = conn.execute(
                "MATCH (a:Author) RETURN a.canonical_id, a.name_variants"
            )
            center_id = "<not_found>"
            name_lower = author_name.lower()
            while res_all.has_next():
                row = res_all.get_next()
                variants = row[1] or []
                if any(name_lower in v.lower() for v in variants):
                    center_id = str(row[0])
                    break
        if center_id == "<not_found>":
            return CoauthorNeighborhoodResult(center_id="<not_found>")
    else:
        raise ValueError("Either author_id or author_name must be provided")

    # ── fetch center node ────────────────────────────────────────────────────
    res = conn.execute(
        "MATCH (a:Author {canonical_id: $cid}) RETURN a.name_variants",
        parameters={"cid": center_id},
    )
    center_variants: list[str] = []
    if res.has_next():
        center_variants = list(res.get_next()[0] or [])

    result = CoauthorNeighborhoodResult(center_id=center_id)
    result.nodes.append(
        CoauthorNode(canonical_id=center_id, name_variants=center_variants, hop=0)
    )

    seen_ids: set[str] = {center_id}

    # ── 1-hop ────────────────────────────────────────────────────────────────
    res1 = conn.execute(
        """MATCH (center:Author {canonical_id: $cid})-[e:COAUTHORED_WITH]->(n:Author)
           RETURN n.canonical_id, n.name_variants, e.paper_id, e.year
           LIMIT $lim
        """,
        parameters={"cid": center_id, "lim": max_nodes},
    )
    hop1_ids: list[str] = []
    while res1.has_next():
        row = res1.get_next()
        nid = str(row[0])
        if nid not in seen_ids:
            seen_ids.add(nid)
            result.nodes.append(
                CoauthorNode(
                    canonical_id=nid,
                    name_variants=list(row[1] or []),
                    hop=1,
                )
            )
            hop1_ids.append(nid)
        result.edges.append(
            CoauthorEdge(src=center_id, dst=nid, paper_id=str(row[2]), year=int(row[3]))
        )

    if hops == 2:
        remaining = max_nodes - len(hop1_ids)
        if remaining > 0 and hop1_ids:
            # For each hop-1 node, fetch its neighbours (excluding already-seen)
            for h1_id in hop1_ids[:10]:  # cap to avoid N+1 explosion
                res2 = conn.execute(
                    """MATCH (n:Author {canonical_id: $h1id})-[e:COAUTHORED_WITH]->(m:Author)
                       RETURN m.canonical_id, m.name_variants, e.paper_id, e.year
                       LIMIT $lim
                    """,
                    parameters={"h1id": h1_id, "lim": max(1, remaining // len(hop1_ids[:10]))},
                )
                while res2.has_next():
                    row = res2.get_next()
                    nid = str(row[0])
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        result.nodes.append(
                            CoauthorNode(
                                canonical_id=nid,
                                name_variants=list(row[1] or []),
                                hop=2,
                            )
                        )
                    result.edges.append(
                        CoauthorEdge(
                            src=h1_id, dst=nid, paper_id=str(row[2]), year=int(row[3])
                        )
                    )

    return result
