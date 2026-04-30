# src/analytics/

**Purpose:** Analytics layer — both the v1 re-export shim and the v2 analytics extensions.

## Contents

| File | Role |
|------|------|
| `__init__.py` | Re-exports `AnalyticsEngine` from `src.v1.analytics` for backwards compatibility |

## v2 plan

New graph-based analytics modules (trend queries, collaboration heatmaps, GNN forecasting) will be added here as siblings. The authoritative v2 analytics architecture lives in [`src/v2/analytics/`](../v2/analytics/).

See [`coursework_v2/DEV_PLAN.md §3`](../../coursework_v2/DEV_PLAN.md) for the full design.
