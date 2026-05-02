"""
LangGraph coordinator — top-level StateGraph[WorkItem].

Graph topology
--------------
::

    START
      │
      ▼
    [source_resolve]   — SourceRouter: query → CanonicalPaper + PDF path
      │
      ▼
    [ensemble_parse]   — parse_with_ensemble: PDF → ParsedDoc + DisagreementSet
      │
      ▼
    [reflexion_fetch]  — ReflexionStore: inject per-venue verbal memory
      │
      ▼
    [plan]             — Planner: CanonicalPaper + layout + reflexion → Plan
      │
      ▼
    [extract]          — Specialist extractors (header, footnote, ack, email)
      │
      ▼
    [merge]            — merge.py: combine specialists → merged_candidates
      │
      ▼
    [critic]           — Critic: tool-grounded verifier → Verdicts + salvage
      │  ╲
      │   retry_count < 2 AND has_retry_verdicts → back to [extract]
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

    graph = await build_graph()
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

_MAX_RETRIES = 2


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
        "_layout_sig": layout_sig,
    }


@_timed("reflexion_fetch")
async def _node_reflexion_fetch(state: WorkItem) -> dict[str, Any]:
    """Fetch per-venue verbal memory from ReflexionStore and inject into WorkItem."""
    from .reflexion import ReflexionStore

    paper = state.canonical_paper
    if paper is None:
        return {}  # no memory to inject

    store = ReflexionStore()
    layout_hash = _layout_hash(state)
    memory = await store.fetch(
        venue=paper.venue or "",
        year=paper.year,
        layout_hash=layout_hash,
    )
    if memory:
        _LOG.debug("reflexion_fetch.hit", venue=paper.venue, memory_len=len(memory))
    return {"reflexion_memory": memory}


@_timed("plan")
async def _node_plan(state: WorkItem) -> dict[str, Any]:
    """Invoke Planner to produce a Plan."""
    from .planner import Planner

    paper = state.canonical_paper
    if paper is None:
        paper = CanonicalPaper(paper_id="unknown", title=state.query, source="unknown")

    planner = Planner()
    plan = await planner.plan(
        paper,
        layout_signature=None,
        reflexion_memory=state.reflexion_memory,
    )

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

    # On retry, only re-run specialists that the Critic flagged
    if state.retry_count > 0 and state.critic_retry_hints:
        extractors_to_run = [e for e in extractors_to_run if e in state.critic_retry_hints]

    header_cands: list[dict[str, Any]] = []
    footnote_cands: list[dict[str, Any]] = []
    ack_cands: list[dict[str, Any]] = []
    email_cands: list[dict[str, Any]] = []
    token_usage: dict[str, dict[str, int]] = {}

    if "header" in extractors_to_run:
        from src.v2.agents.extractors.header import HeaderExtractor
        retry_hint = state.critic_retry_hints.get("header", "")
        hx = HeaderExtractor()
        cands = await hx.extract(doc, retry_hint=retry_hint)
        header_cands = [c.model_dump(mode="json") for c in cands.items]
        token_usage["header"] = cands.token_usage

    if "footnote" in extractors_to_run:
        from src.v2.agents.extractors.footnote import FootnoteExtractor
        retry_hint = state.critic_retry_hints.get("footnote", "")
        fx = FootnoteExtractor()
        cands = await fx.extract(doc, retry_hint=retry_hint)
        footnote_cands = [c.model_dump(mode="json") for c in cands.items]
        token_usage["footnote"] = cands.token_usage

    if "acknowledgements" in extractors_to_run:
        from src.v2.agents.extractors.acknowledgements import AcknowledgementsExtractor
        retry_hint = state.critic_retry_hints.get("acknowledgements", "")
        ax = AcknowledgementsExtractor()
        cands = await ax.extract(doc, retry_hint=retry_hint)
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

    # Merge retry results back into existing candidates (keep non-retried specialists)
    if state.retry_count > 0:
        if "header" not in extractors_to_run:
            header_cands = list(state.header_candidates)
        if "footnote" not in extractors_to_run:
            footnote_cands = list(state.footnote_candidates)
        if "acknowledgements" not in extractors_to_run:
            ack_cands = list(state.ack_candidates)
        if "email_domain" not in extractors_to_run:
            email_cands = list(state.email_candidates)

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


@_timed("critic")
async def _node_critic(state: WorkItem) -> dict[str, Any]:
    """Run the Critic: collect evidence, judge candidates, apply salvage."""
    from src.v2.agents.critic.critic import Critic
    from src.v2.orchestration.contracts import Candidate

    paper = state.canonical_paper
    if paper is None:
        paper = CanonicalPaper(paper_id="unknown", title=state.query, source="unknown")

    candidates = [
        Candidate.model_validate(c) for c in state.merged_candidates
    ]

    critic = Critic()
    verdicts = await critic.judge(candidates, paper)

    # Build retry hints for the next extract pass
    retry_hints: dict[str, str] = {}
    for v in verdicts:
        if v.decision == "retry" and v.retry_hint:
            # Map retry hint to the candidate's source_specialist
            cand_map = {c.candidate_id: c for c in candidates}
            cand = cand_map.get(v.candidate_id)
            if cand:
                retry_hints[cand.source_specialist] = v.retry_hint

    has_retries = bool(retry_hints) and state.retry_count < _MAX_RETRIES

    return {
        "verdicts": [v.model_dump(mode="json") for v in verdicts],
        "critic_retry_hints": retry_hints,
        "retry_count": state.retry_count + (1 if has_retries else 0),
        "status": "complete" if not has_retries else "running",
    }


@_timed("reflexion_write")
async def _node_reflexion_write(state: WorkItem) -> dict[str, Any]:
    """Persist reflexion memory after the Critic has finished (final pass only)."""
    from .reflexion import ReflexionStore

    paper = state.canonical_paper
    if paper is None or state.status != "complete":
        return {}

    accepted = [
        c for c, v in zip(
            state.merged_candidates,
            state.verdicts,
        )
        if isinstance(v, dict) and v.get("decision") == "accept"
    ]
    store = ReflexionStore()
    layout_hash = _layout_hash(state)
    try:
        await store.reflect(
            paper_id=paper.paper_id,
            venue=paper.venue or "",
            year=paper.year,
            layout_hash=layout_hash,
            accepted_candidates=accepted,
            baseline_candidates=state.merged_candidates,
        )
    except Exception as exc:
        _LOG.warning("reflexion_write.failed", error=str(exc))
    return {}


# ─────────────────────────── routing ────────────────────────────────────────


def _after_critic(state: WorkItem) -> str:
    """Route back to extract if retries are pending, otherwise reflexion_write."""
    if state.status == "failed":
        return "reflexion_write"
    if state.retry_count > 0 and state.critic_retry_hints and state.retry_count <= _MAX_RETRIES:
        # Only retry if the count was just incremented (status still "running")
        if state.status == "running":
            return "extract"
    return "reflexion_write"


# ─────────────────────────── helpers ────────────────────────────────────────


def _layout_hash(state: WorkItem) -> str | None:
    """Derive a short layout hash from the disagreement_set, if available."""
    import hashlib
    if not state.disagreement_set:
        return None
    raw = str(state.disagreement_set.get("span_counts", ""))
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


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
    graph.add_node("reflexion_fetch", _node_reflexion_fetch)  # type: ignore[call-overload]
    graph.add_node("plan", _node_plan)  # type: ignore[call-overload]
    graph.add_node("extract", _node_extract)  # type: ignore[call-overload]
    graph.add_node("merge", _node_merge)  # type: ignore[call-overload]
    graph.add_node("critic", _node_critic)  # type: ignore[call-overload]
    graph.add_node("reflexion_write", _node_reflexion_write)  # type: ignore[call-overload]

    # Edges
    graph.add_edge(START, "source_resolve")
    graph.add_conditional_edges(
        "source_resolve",
        lambda s: END if s.status == "failed" else "ensemble_parse",
        {"ensemble_parse": "ensemble_parse", END: END},
    )
    graph.add_conditional_edges(
        "ensemble_parse",
        lambda s: END if s.status == "failed" else "reflexion_fetch",
        {"reflexion_fetch": "reflexion_fetch", END: END},
    )
    graph.add_edge("reflexion_fetch", "plan")
    graph.add_edge("plan", "extract")
    graph.add_conditional_edges(
        "extract",
        lambda s: END if s.status == "failed" else "merge",
        {"merge": "merge", END: END},
    )
    graph.add_edge("merge", "critic")
    graph.add_conditional_edges(
        "critic",
        _after_critic,
        {"extract": "extract", "reflexion_write": "reflexion_write"},
    )
    graph.add_edge("reflexion_write", END)

    if use_checkpointer:
        checkpointer = await make_checkpointer()
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
