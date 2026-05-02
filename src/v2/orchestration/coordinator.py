"""
LangGraph coordinator — top-level StateGraph[WorkItem].

Graph topology (skeleton)
-------------------------
::

    START
      │
      ▼
    [source_resolve]  — SourceRouter: query → CanonicalPaper + PDF path
      │
      ▼
    [ensemble_parse]  — parse_with_ensemble: PDF → ParsedDoc + DisagreementSet
      │
      ▼
    [plan]            — Planner: CanonicalPaper + layout → Plan
      │
      ▼
    [extract]         — Specialist extractors (header, footnote, ack, email)
      │
      ▼
    [merge]           — merge.py: combine specialists → merged_candidates
      │
      ▼
    [no_op_verify]    — placeholder for Stage 5 Critic
      │
      ▼
    END

Each node:
  * logs step_started / step_completed with timing via structlog
  * stores timings in WorkItem.node_timings
  * on any exception sets WorkItem.status="failed" and re-raises

Usage
-----
::

    from src.v2.orchestration.coordinator import build_graph

    graph = build_graph()
    result = await graph.ainvoke(
        WorkItem(query="cat:cs.AI"),
        config={"configurable": {"thread_id": "run-001"}},
    )
"""
from __future__ import annotations

import functools
import time
from collections.abc import Callable, Coroutine
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from .checkpointing import make_checkpointer
from .contracts import CanonicalPaper, Plan, WorkItem

_LOG = structlog.get_logger(__name__)


# ─────────────────────────── node helpers ────────────────────────────────────


_NodeFn = Callable[[WorkItem], Coroutine[Any, Any, dict[str, Any]]]


def _timed(node_name: str) -> Callable[[_NodeFn], _NodeFn]:
    """Decorator that adds timing and structured logging to a node function."""

    def decorator(fn: _NodeFn) -> _NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: WorkItem) -> dict[str, Any]:
            _LOG.info("step_started", node=node_name, work_id=state.work_id)
            t0 = time.perf_counter()
            try:
                result: dict[str, Any] = await fn(state)
                elapsed = time.perf_counter() - t0
                timings = {**state.node_timings, node_name: round(elapsed, 3)}
                result["node_timings"] = timings
                _LOG.info("step_completed", node=node_name, elapsed_s=round(elapsed, 3))
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                _LOG.error("step_failed", node=node_name, error=str(exc), elapsed_s=round(elapsed, 3))
                return {
                    "status": "failed",
                    "error": f"{node_name}: {exc}",
                    "node_timings": {**state.node_timings, node_name: round(elapsed, 3)},
                }

        return wrapper

    return decorator


# ─────────────────────────── nodes ──────────────────────────────────────────


@_timed("source_resolve")
async def _node_source_resolve(state: WorkItem) -> dict[str, Any]:
    """Resolve query → CanonicalPaper + PDF path."""
    from .source_router import SourceRouter

    router = SourceRouter()
    results = await router.resolve(state.query, n=1)
    if not results:
        return {"status": "failed", "error": "source_resolve: no paper found"}

    paper, pdf_path = results[0]
    return {
        "canonical_paper": paper,
        "pdf_path": str(pdf_path),
        "status": "running",
    }


@_timed("ensemble_parse")
async def _node_ensemble_parse(state: WorkItem) -> dict[str, Any]:
    """Run parser ensemble on the downloaded PDF."""
    import os
    from pathlib import Path

    from src.v2.parsers import parse_with_ensemble

    if not state.pdf_path:
        return {"status": "failed", "error": "ensemble_parse: no pdf_path"}

    skip = os.getenv("SKIP_HEAVY_PARSERS", "")
    parsers = ("pymupdf",) if skip.lower() in ("1", "true", "yes") else ("docling", "marker", "pymupdf")

    doc, ds = parse_with_ensemble(Path(state.pdf_path), parsers=parsers)

    layout_sig: dict[str, Any] = {
        "parsers_used": ds.parsers_used,
        "span_counts": ds.span_counts,
        "has_disagreements": not ds.is_empty,
        "disagreement_count": len(ds.disagreements),
    }

    return {
        "parsed_doc": doc.model_dump(mode="json"),
        "disagreement_set": ds.model_dump(mode="json"),
        "_layout_sig": layout_sig,  # ephemeral; passed to planner below
    }


@_timed("plan")
async def _node_plan(state: WorkItem) -> dict[str, Any]:
    """Invoke Planner to produce a Plan."""
    from .planner import Planner

    paper = state.canonical_paper
    if paper is None:
        paper = CanonicalPaper(paper_id="unknown", title=state.query, source="unknown")

    planner = Planner()
    plan = await planner.plan(paper, layout_signature=None)

    return {"plan": plan}


@_timed("extract")
async def _node_extract(state: WorkItem) -> dict[str, Any]:
    """Run specialist extractors according to the Plan."""
    from pathlib import Path

    from src.v2.parsers.schemas import ParsedDoc

    if not state.parsed_doc:
        return {"status": "failed", "error": "extract: no parsed_doc"}

    plan: Plan = state.plan or Plan(
        parsers=["pymupdf"],
        extractors=["header", "email_domain"],
        verification_depth="light",
        reasons="fallback",
    )

    doc = ParsedDoc.model_validate(state.parsed_doc)
    extractors_to_run = plan.extractors

    header_cands: list[dict[str, Any]] = []
    footnote_cands: list[dict[str, Any]] = []
    ack_cands: list[dict[str, Any]] = []
    email_cands: list[dict[str, Any]] = []
    token_usage: dict[str, dict[str, int]] = {}

    if "header" in extractors_to_run:
        from src.v2.agents.extractors.header import HeaderExtractor
        hx = HeaderExtractor()
        cands = await hx.extract(doc)
        header_cands = [c.model_dump(mode="json") for c in cands.items]
        token_usage["header"] = cands.token_usage

    if "footnote" in extractors_to_run:
        from src.v2.agents.extractors.footnote import FootnoteExtractor
        fx = FootnoteExtractor()
        cands = await fx.extract(doc)
        footnote_cands = [c.model_dump(mode="json") for c in cands.items]
        token_usage["footnote"] = cands.token_usage

    if "acknowledgements" in extractors_to_run:
        from src.v2.agents.extractors.acknowledgements import AcknowledgementsExtractor
        ax = AcknowledgementsExtractor()
        cands = await ax.extract(doc)
        ack_cands = [c.model_dump(mode="json") for c in cands.items]
        token_usage["acknowledgements"] = cands.token_usage

    if "email_domain" in extractors_to_run:
        from src.v2.agents.extractors.email_domain import EmailDomainExtractor
        ex = EmailDomainExtractor()
        cands = await ex.extract(doc)
        email_cands = [c.model_dump(mode="json") for c in cands.items]
        token_usage["email_domain"] = cands.token_usage

    existing_token_usage = dict(state.token_usage)
    existing_token_usage.update(token_usage)

    return {
        "header_candidates": header_cands,
        "footnote_candidates": footnote_cands,
        "ack_candidates": ack_cands,
        "email_candidates": email_cands,
        "token_usage": existing_token_usage,
    }


@_timed("merge")
async def _node_merge(state: WorkItem) -> dict[str, Any]:
    """Merge specialist outputs into deduplicated Candidates."""
    from src.v2.agents.extractors.merge import merge_candidates

    merged = merge_candidates(
        header=state.header_candidates,
        footnote=state.footnote_candidates,
        ack=state.ack_candidates,
        email=state.email_candidates,
    )
    return {"merged_candidates": [c.model_dump(mode="json") for c in merged]}


@_timed("verify")
async def _node_verify_noop(state: WorkItem) -> dict[str, Any]:
    """No-op verification placeholder (Stage 5 will replace this)."""
    _LOG.info("verify.noop", candidates=len(state.merged_candidates))
    return {"status": "complete", "verdicts": []}


# ─────────────────────────── routing ────────────────────────────────────────


def _should_continue(state: WorkItem) -> str:
    if state.status == "failed":
        return END
    return "continue"


# ─────────────────────────── graph factory ──────────────────────────────────


async def build_graph(use_checkpointer: bool = True) -> Any:  # CompiledGraph
    """Build and compile the coordinator StateGraph.

    Parameters
    ----------
    use_checkpointer:
        If True, attach the async SQLite checkpointer for resumable runs.
    """
    graph: StateGraph = StateGraph(WorkItem)  # type: ignore[type-arg]

    # Register nodes
    graph.add_node("source_resolve", _node_source_resolve)  # type: ignore[call-overload]
    graph.add_node("ensemble_parse", _node_ensemble_parse)  # type: ignore[call-overload]
    graph.add_node("plan", _node_plan)  # type: ignore[call-overload]
    graph.add_node("extract", _node_extract)  # type: ignore[call-overload]
    graph.add_node("merge", _node_merge)  # type: ignore[call-overload]
    graph.add_node("verify", _node_verify_noop)  # type: ignore[call-overload]

    # Edges
    graph.add_edge(START, "source_resolve")

    # Conditional: bail on failure after source_resolve
    graph.add_conditional_edges(
        "source_resolve",
        lambda s: END if s.status == "failed" else "ensemble_parse",
        {"ensemble_parse": "ensemble_parse", END: END},
    )
    graph.add_conditional_edges(
        "ensemble_parse",
        lambda s: END if s.status == "failed" else "plan",
        {"plan": "plan", END: END},
    )
    graph.add_edge("plan", "extract")
    graph.add_conditional_edges(
        "extract",
        lambda s: END if s.status == "failed" else "merge",
        {"merge": "merge", END: END},
    )
    graph.add_edge("merge", "verify")
    graph.add_edge("verify", END)

    if use_checkpointer:
        checkpointer = await make_checkpointer()
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
