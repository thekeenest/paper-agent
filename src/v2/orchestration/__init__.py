"""
Orchestration layer — Coordinator, Planner, Source Router, Checkpointing.

Public surface
--------------
::

    from src.v2.orchestration import build_graph, WorkItem, Plan, CanonicalPaper
    from src.v2.orchestration.source_router import SourceRouter
    from src.v2.orchestration.planner import Planner
    from src.v2.orchestration.checkpointing import make_checkpointer

Components
----------
  contracts.py      — shared Pydantic v2 models (WorkItem, Plan, Verdict, …)
  coordinator.py    — LangGraph StateGraph[WorkItem]; wires all nodes
  planner.py        — ChatOpenAI node → Plan(parsers, extractors, depth)
  source_router.py  — ArXiv/OpenAlex/S2/ACL/OpenReview → CanonicalPaper + PDF
  checkpointing.py  — SQLite-backed LangGraph checkpointer

See DEV_PLAN.md §3.2 and docs/architecture.md for the full design.
"""

from .contracts import CanonicalPaper, Candidate, Candidates, EvidenceTrail, Plan, ToolEvidence, Verdict, WorkItem
from .coordinator import build_graph

__all__ = [
    "build_graph",
    "WorkItem",
    "Plan",
    "Verdict",
    "ToolEvidence",
    "Candidate",
    "Candidates",
    "EvidenceTrail",
    "CanonicalPaper",
]
