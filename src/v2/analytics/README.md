# src/v2/analytics/

**Purpose:** Graph-based analytics over the temporal KuzuDB knowledge graph.

Supersedes v1's `AnalyticsEngine` (CSV → `value_counts()` → matplotlib) with
proper graph queries and optional GNN forecasting.

## Planned modules

| Module | Output |
|---|---|
| `trend_analytics.py` | Industry/academia split per venue per year |
| `country_analytics.py` | Country-level influence heatmaps |
| `collaboration.py` | Weighted co-author network metrics |
| `gnn_forecasting.py` | 1-year collaboration link-prediction (DHGNN-lite) |
| `graphrag_demo.py` | NL Q&A over KG for the coursework demo |

## Relationship to v1

`from src.analytics import AnalyticsEngine` still works via the v1 re-export shim
at `src/analytics/__init__.py`. This package is additive, not a replacement.

See [DEV_PLAN.md §3.6](../../../coursework_v2/DEV_PLAN.md) and
[READING_LIST.md §F (bibliometrics, DHGNN)](../../../coursework_v2/READING_LIST.md).
