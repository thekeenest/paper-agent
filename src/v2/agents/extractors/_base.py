"""
Base class for all specialist extractor agents.

Each specialist:
  1. Receives a subset of ParsedDoc spans (only its region).
  2. Passes them to a focused LLM prompt.
  3. Post-validates the output (rejects trivially wrong names, etc.).
  4. Returns Candidates with per-span evidence IDs.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.v2.orchestration.contracts import Candidate, Candidates, EvidenceTrail

_LOG = structlog.get_logger(__name__)

_DEFAULT_MODEL = os.getenv("EXTRACTOR_MODEL", "gpt-4o-mini")
_MAX_SPANS = 60  # hard cap: never send more than this many spans to one LLM call


def _span_id(text: str, page: int, specialist: str) -> str:
    """Stable deterministic ID for a span (page + text hash)."""
    h = hashlib.sha1(f"{specialist}|{page}|{text}".encode()).hexdigest()[:10]
    return f"span_{h}"


class _RawCandidate(BaseModel):
    """Intermediate schema the LLM must produce."""

    author_name: str = Field(description="Full author name as printed in the paper")
    affiliations: list[str] = Field(default_factory=list, description="Institution strings")
    emails: list[str] = Field(default_factory=list, description="Email addresses for this author")


class _LLMOutput(BaseModel):
    """The structured output schema every extractor LLM must return."""

    candidates: list[_RawCandidate]


class BaseExtractor:
    """Shared plumbing for all specialist extractors.

    Sub-classes must set:
      * ``_SYSTEM_PROMPT``  — focused instruction string
      * ``_SPECIALIST``     — name used in logging and Candidate.source_specialist
    """

    _SYSTEM_PROMPT: str = ""
    _SPECIALIST: str = "base"

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        m = model or _DEFAULT_MODEL
        llm = ChatOpenAI(model=m, temperature=temperature)
        self._chain = llm.with_structured_output(_LLMOutput)

    def _build_messages(self, span_text: str, retry_hint: str = "") -> list[Any]:
        content = span_text
        if retry_hint:
            content = f"{span_text}\n\n[Critic retry hint]: {retry_hint}"
        return [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

    async def _call_llm(self, span_text: str) -> _LLMOutput:
        result = await self._chain.ainvoke(self._build_messages(span_text))
        if isinstance(result, _LLMOutput):
            return result
        _LOG.warning("extractor.unexpected_output_type", specialist=self._SPECIALIST)
        return _LLMOutput(candidates=[])

    def _to_candidates(
        self,
        raw: _LLMOutput,
        span_ids: list[str],
        token_usage: dict[str, int],
    ) -> Candidates:
        items: list[Candidate] = []
        for rc in raw.candidates:
            try:
                c = Candidate(
                    author_name=rc.author_name,
                    affiliations=rc.affiliations,
                    emails=rc.emails,
                    source_specialist=self._SPECIALIST,
                    evidence_span_ids=span_ids,
                    confidence="medium",
                    evidence_trail=EvidenceTrail(),
                )
                items.append(c)
            except (ValueError, Exception) as exc:
                _LOG.warning(
                    "extractor.candidate_rejected",
                    specialist=self._SPECIALIST,
                    name=rc.author_name,
                    reason=str(exc),
                )
        return Candidates(items=items, specialist=self._SPECIALIST, token_usage=token_usage)
