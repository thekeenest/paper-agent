# src/v2/orchestration/

**Purpose:** Top-level LangGraph orchestration — Coordinator, Planner, and Reflexion controller.

Planned modules:
- `coordinator.py` — root StateGraph; routes tasks to parser and extractor sub-graphs
- `planner.py` — decides parser order, verification depth, and source priority per paper
- `reflexion.py` — verbal-memory controller; logs per-venue mistakes and updates policy

See [DEV_PLAN.md §3.2](../../../coursework_v2/DEV_PLAN.md) for the full architecture.
