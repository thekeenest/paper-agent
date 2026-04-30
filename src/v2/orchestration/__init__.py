"""
Orchestration layer — Coordinator, Planner, and Reflexion controller.

Planned components (see DEV_PLAN.md §3.2):
  Coordinator   — top-level LangGraph StateGraph; dispatches to Planner and sub-graphs
  Planner       — selects parser order, verification depth, source priority
  ReflexionCtrl — maintains per-venue verbal memory; updates extraction policy

Status: STUB — no implementation yet.
"""
