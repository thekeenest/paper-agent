"""
Query: country_strategy_heatmap
--------------------------------
Return a pairwise collaboration count matrix: how many papers have at least one
author from country_a and at least one author from country_b, broken down by year.

Result is suitable for rendering as a heatmap (country_a × country_b).

Return type
-----------
CountryHeatmapResult — list of CountryPairRow.

Cypher strategy
---------------
Walk COLLABORATED_WITH edges between institutions, joining through country_code.
Because COLLABORATED_WITH carries ``year`` and ``papers_count``, we aggregate
directly from the edge without re-scanning Paper nodes.

Usage
-----
::

    from src.v2.kg.queries.country_strategy_heatmap import country_strategy_heatmap

    result = country_strategy_heatmap(conn, year=2023)
    for row in result.rows:
        print(row.country_a, row.country_b, row.year, row.paper_count)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CountryPairRow:
    """Collaboration count for one (country_a, country_b, year) triple."""

    country_a: str
    country_b: str
    year: int
    paper_count: int


@dataclass
class CountryHeatmapResult:
    """Container returned by :func:`country_strategy_heatmap`."""

    rows: list[CountryPairRow] = field(default_factory=list)

    def to_matrix(self) -> dict[tuple[str, str], int]:
        """Return a flat dict keyed by (country_a, country_b) summed across years."""
        matrix: dict[tuple[str, str], int] = {}
        for r in self.rows:
            key = (r.country_a, r.country_b)
            matrix[key] = matrix.get(key, 0) + r.paper_count
        return matrix

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "country_a": r.country_a,
                "country_b": r.country_b,
                "year": r.year,
                "paper_count": r.paper_count,
            }
            for r in self.rows
        ]


def country_strategy_heatmap(
    conn: Any,
    year: int | None = None,
    min_papers: int = 1,
) -> CountryHeatmapResult:
    """Return country-pair collaboration counts.

    Parameters
    ----------
    conn:
        Open ``kuzu.Connection``.
    year:
        Restrict to a specific year.  ``None`` returns all years.
    min_papers:
        Only include pairs with at least this many joint papers.

    Returns
    -------
    CountryHeatmapResult
    """
    where_clauses: list[str] = [
        "i_a.country_code <> '' AND i_b.country_code <> ''",
        "i_a.country_code < i_b.country_code",  # canonical ordering to deduplicate
        "collab.papers_count >= $min_papers",
    ]
    params: dict[str, Any] = {"min_papers": min_papers}

    if year is not None:
        where_clauses.append("collab.year = $year")
        params["year"] = year

    where_sql = "WHERE " + " AND ".join(where_clauses)

    cypher = f"""
        MATCH (i_a:Institution)-[collab:COLLABORATED_WITH]->(i_b:Institution)
        {where_sql}
        RETURN
            i_a.country_code  AS country_a,
            i_b.country_code  AS country_b,
            collab.year       AS yr,
            sum(collab.papers_count) AS paper_count
        ORDER BY paper_count DESC
    """

    res = conn.execute(cypher, parameters=params)
    rows: list[CountryPairRow] = []
    while res.has_next():
        row = res.get_next()
        rows.append(
            CountryPairRow(
                country_a=str(row[0]),
                country_b=str(row[1]),
                year=int(row[2]),
                paper_count=int(row[3]),
            )
        )
    return CountryHeatmapResult(rows=rows)
