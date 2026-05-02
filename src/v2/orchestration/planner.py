"""
Planner agent — selects parsers, extractors, and verification depth.

The Planner is a ChatOpenAI node with structured output (Plan).  It is
*deliberately conservative*: its only job is to choose tools, not to extract
any information from the paper.

Inputs
------
* CanonicalPaper    — title, abstract, venue, year
* layout_signature  — span counts / disagreement flags from the ensemble
* reflexion_memory  — per-venue verbal memory (passed in, ignored for now)

Output
------
* Plan(parsers, extractors, verification_depth, reasons)

Usage
-----
::

    from src.v2.orchestration.planner import Planner

    planner = Planner()
    plan = await planner.plan(paper, layout_signature={})
"""
from __future__ import annotations

import os
from typing import Any

import structlog
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .contracts import CanonicalPaper, Plan

_LOG = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are the Planner for a scholarly author-affiliation extraction pipeline.

Your ONLY job is to decide which PDF parsers and extraction specialists to run,
and how deeply to verify the results.  You MUST NOT extract author names,
affiliations, or any other information from the paper.

## Available parsers (choose ≥1, list in priority order)
- "docling"  — IBM layout-aware; best for complex multi-column layouts
- "marker"   — fast Markdown converter; good for single-column ArXiv papers
- "pymupdf"  — always-available bbox heuristic; use as fallback

## Available extractors (choose ≥1)
- "header"          — extracts from the title/author block (top of page 0)
- "footnote"        — extracts from footnotes (affiliation superscripts)
- "acknowledgements"— extracts from the acknowledgements section
- "email_domain"    — resolves email addresses to institutions via DNS + ROR

## Verification depth
- "light"    — skip expensive ROR / OpenAlex cross-checks
- "standard" — run ROR + one OpenAlex lookup per candidate (default)
- "deep"     — run all linkers including S2AND deduplication

## Decision rules
1. For ArXiv CS/ML papers: prefer marker + pymupdf; use docling only if
   the layout looks complex (multi-column, heavy tables).
2. Always include "header" and "email_domain" extractors.
3. Add "footnote" when the venue is a major conference (NeurIPS, ICML, CVPR…).
4. Add "acknowledgements" when the abstract mentions funding or multi-affil authors.
5. Use "deep" verification only for high-citation papers or known noisy venues.

Respond only with valid JSON matching the Plan schema.
"""


class _PlannerInput(BaseModel):
    """Structured input fed to the LLM."""

    paper_title: str
    paper_abstract: str
    paper_venue: str | None
    paper_year: int | None
    paper_source: str
    layout_signature: dict[str, Any]
    reflexion_note: str = "(no prior memory for this venue)"


class Planner:
    """LLM-backed planner that produces a Plan given paper metadata.

    Parameters
    ----------
    model:
        OpenAI model name.  Defaults to ``PLANNER_MODEL`` env var or
        ``gpt-4o-mini`` (cheap and fast enough for planning).
    temperature:
        Sampling temperature.  0 for deterministic plans.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        resolved_model: str = model if model is not None else (os.getenv("PLANNER_MODEL") or "gpt-4o-mini")
        _llm = ChatOpenAI(model=resolved_model, temperature=temperature)
        self._chain = _llm.with_structured_output(Plan)

    async def plan(
        self,
        paper: CanonicalPaper,
        layout_signature: dict[str, Any] | None = None,
        reflexion_memory: str | None = None,
    ) -> Plan:
        """Produce a Plan for *paper*.

        Parameters
        ----------
        paper:
            Resolved canonical paper record.
        layout_signature:
            Dict with keys like ``{"span_counts": {...}, "has_disagreements": bool}``.
            Ignored for now but passed to the LLM so future versions can use it.
        reflexion_memory:
            Per-venue verbal memory from the Reflexion controller.  Passed in
            but currently ignored (stubbed for Stage 4).
        """
        inp = _PlannerInput(
            paper_title=paper.title,
            paper_abstract=paper.abstract[:800],  # truncate for token budget
            paper_venue=paper.venue,
            paper_year=paper.year,
            paper_source=paper.source,
            layout_signature=layout_signature or {},
            reflexion_note=reflexion_memory or "(no prior memory for this venue)",
        )

        user_msg = (
            f"Paper: {inp.paper_title!r}\n"
            f"Venue: {inp.paper_venue or 'unknown'} ({inp.paper_year or 'unknown year'})\n"
            f"Source: {inp.paper_source}\n"
            f"Abstract (first 800 chars): {inp.paper_abstract}\n"
            f"Layout signature: {inp.layout_signature}\n"
            f"Venue memory: {inp.reflexion_note}\n\n"
            "Choose the parsers, extractors, and verification depth."
        )

        _LOG.info("planner.start", title=paper.title[:60])

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]

        result = await self._chain.ainvoke(messages)

        # Ensure result is a Plan (with_structured_output should guarantee this,
        # but we be explicit for mypy)
        if not isinstance(result, Plan):
            _LOG.warning("planner.unexpected_type", type=type(result).__name__)
            result = Plan(
                parsers=["pymupdf"],
                extractors=["header", "email_domain"],
                verification_depth="light",
                reasons="Fallback plan (planner returned unexpected type).",
            )

        _LOG.info(
            "planner.done",
            parsers=result.parsers,
            extractors=result.extractors,
            depth=result.verification_depth,
        )
        return result
