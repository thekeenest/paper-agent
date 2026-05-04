"""
Query: top_institutions_by_venue
---------------------------------
For a given venue (and optionally year), return institutions ranked by the
number of accepted-author affiliations they contributed.

Return type
-----------
TopInstitutionsResult — list of InstitutionRankRow.

Cypher strategy
---------------
Walk Author → AUTHORED → Paper → PUBLISHED_AT → Venue and
Author → AFFILIATED_AT → Institution, grouping by institution.

Usage
-----
::

    from src.v2.kg.queries.top_institutions_by_venue import top_institutions_by_venue

    result = top_institutions_by_venue(conn, venue="NeurIPS", year=2024, top_k=10)
    for row in result.rows:
        print(row.rank, row.canonical_name, row.affiliation_count)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InstitutionRankRow:
    """One institution entry in the ranked list."""

    rank: int
    ror_id: str
    canonical_name: str
    org_type: str
    country_code: str
    affiliation_count: int  # number of author-affiliation edges to this institution


@dataclass
class TopInstitutionsResult:
    """Container returned by :func:`top_institutions_by_venue`."""

    venue_key: str
    year: int | None
    rows: list[InstitutionRankRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_key": self.venue_key,
            "year": self.year,
            "rows": [
                {
                    "rank": r.rank,
                    "ror_id": r.ror_id,
                    "canonical_name": r.canonical_name,
                    "org_type": r.org_type,
                    "country_code": r.country_code,
                    "affiliation_count": r.affiliation_count,
                }
                for r in self.rows
            ],
        }


def top_institutions_by_venue(
    conn: Any,
    venue: str,
    year: int | None = None,
    top_k: int = 20,
) -> TopInstitutionsResult:
    """Return top-k institutions by affiliation count for a venue.

    Parameters
    ----------
    conn:
        Open ``kuzu.Connection``.
    venue:
        Venue name or venue_key (matched via ``CONTAINS``).
    year:
        Filter to specific year.  ``None`` aggregates across all years.
    top_k:
        Maximum number of rows to return.

    Returns
    -------
    TopInstitutionsResult
    """
    import re

    venue_key = re.sub(r"\s+", "_", venue.strip().lower()) or "unknown"

    where_clauses = ["LOWER(v.key) CONTAINS $vkey"]
    params: dict[str, Any] = {"vkey": venue_key, "top_k": top_k}

    if year is not None:
        where_clauses.append("pub.year = $year")
        params["year"] = year

    where_sql = "WHERE " + " AND ".join(where_clauses)

    cypher = f"""
        MATCH (a:Author)-[aff:AFFILIATED_AT]->(i:Institution)
        MATCH (a)-[:AUTHORED]->(p:Paper)-[pub:PUBLISHED_AT]->(v:Venue)
        {where_sql}
        RETURN
            i.ror_id          AS ror_id,
            i.canonical_name  AS canonical_name,
            i.org_type        AS org_type,
            i.country_code    AS country_code,
            count(*)          AS aff_count
        ORDER BY aff_count DESC, ror_id
        LIMIT $top_k
    """

    res = conn.execute(cypher, parameters=params)
    rows: list[InstitutionRankRow] = []
    rank = 1
    while res.has_next():
        row = res.get_next()
        rows.append(
            InstitutionRankRow(
                rank=rank,
                ror_id=str(row[0]),
                canonical_name=str(row[1]),
                org_type=str(row[2]),
                country_code=str(row[3]),
                affiliation_count=int(row[4]),
            )
        )
        rank += 1

    return TopInstitutionsResult(venue_key=venue_key, year=year, rows=rows)
