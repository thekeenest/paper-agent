"""
Query: industry_share_by_venue
------------------------------
For each (venue, year) pair return the fraction of accepted author-affiliations
that belong to industry organisations (org_type == "company").

Return type
-----------
IndustryShareResult — a list of IndustryShareRow, one per (venue, year).

Cypher strategy
---------------
Walk Paper → PUBLISHED_AT → Venue, then Author → AFFILIATED_AT → Institution.
Count total affiliations and industry affiliations per (venue, year).

Usage
-----
::

    from src.v2.kg.schema import open_db, KGSchema
    from src.v2.kg.queries.industry_share_by_venue import industry_share_by_venue

    db, conn = open_db()
    KGSchema.create_all(conn)
    result = industry_share_by_venue(conn, venue="NeurIPS", year=2024)
    for row in result.rows:
        print(row.venue_key, row.year, row.industry_share)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IndustryShareRow:
    """One (venue_key, year) data point."""

    venue_key: str
    year: int
    total_affiliations: int
    industry_affiliations: int
    industry_share: float  # industry_affiliations / total_affiliations, or 0.0


@dataclass
class IndustryShareResult:
    """Container returned by :func:`industry_share_by_venue`."""

    rows: list[IndustryShareRow] = field(default_factory=list)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "venue_key": r.venue_key,
                "year": r.year,
                "total_affiliations": r.total_affiliations,
                "industry_affiliations": r.industry_affiliations,
                "industry_share": round(r.industry_share, 4),
            }
            for r in self.rows
        ]


def industry_share_by_venue(
    conn: Any,
    venue: str | None = None,
    year: int | None = None,
) -> IndustryShareResult:
    """Return industry-affiliation share for each (venue, year).

    Parameters
    ----------
    conn:
        Open ``kuzu.Connection``.
    venue:
        Filter to a specific venue (matched against venue_key with ``CONTAINS``).
        If ``None`` all venues are returned.
    year:
        Filter to a specific year.  If ``None`` all years are returned.

    Returns
    -------
    IndustryShareResult
    """
    # Build a parameterised query.  KuzuDB supports $param syntax.
    where_clauses: list[str] = []
    params: dict[str, Any] = {}

    if venue is not None:
        where_clauses.append("LOWER(v.key) CONTAINS LOWER($venue)")
        params["venue"] = venue
    if year is not None:
        where_clauses.append("pub.year = $year")
        params["year"] = year

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    cypher = f"""
        MATCH (a:Author)-[aff:AFFILIATED_AT]->(i:Institution)
        MATCH (a)-[:AUTHORED]->(p:Paper)-[pub:PUBLISHED_AT]->(v:Venue)
        {where_sql}
        RETURN
            v.key                                    AS venue_key,
            pub.year                                 AS yr,
            count(*)                                 AS total,
            sum(CASE WHEN i.org_type = 'company' THEN 1 ELSE 0 END) AS industry
        ORDER BY venue_key, yr
    """

    res = conn.execute(cypher, parameters=params)
    rows: list[IndustryShareRow] = []
    while res.has_next():
        row = res.get_next()
        venue_key = str(row[0])
        yr = int(row[1])
        total = int(row[2])
        ind = int(row[3])
        share = ind / total if total > 0 else 0.0
        rows.append(
            IndustryShareRow(
                venue_key=venue_key,
                year=yr,
                total_affiliations=total,
                industry_affiliations=ind,
                industry_share=share,
            )
        )
    return IndustryShareResult(rows=rows)
