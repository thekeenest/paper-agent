"""
Orchestration data contracts — shared Pydantic v2 models.

These models form the strict type boundary between every LangGraph node.
Any node that mutates state must produce a value whose type matches these
definitions; LangGraph will raise at the boundary if it doesn't.

Public surface
--------------
::

    from src.v2.orchestration.contracts import (
        WorkItem, Plan, Verdict, ToolEvidence, Candidate, EvidenceTrail,
        CanonicalPaper,
    )
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ─────────────────────────── provenance / evidence ───────────────────────────


class ToolEvidence(BaseModel):
    """A single piece of evidence produced by one linker / tool call."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool: str = Field(description="Tool name, e.g. 'ror_linker', 'openalex', 's2'")
    query: str
    result_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvidenceTrail(BaseModel):
    """Ordered list of evidence items that support a single Candidate."""

    model_config = ConfigDict(frozen=True)

    items: list[ToolEvidence] = Field(default_factory=list)

    def add(self, item: ToolEvidence) -> EvidenceTrail:
        return EvidenceTrail(items=[*self.items, item])

    @property
    def sources(self) -> list[str]:
        return [e.tool for e in self.items]

    @property
    def max_confidence(self) -> float:
        if not self.items:
            return 0.0
        return max(e.confidence for e in self.items)


# ─────────────────────────── extraction outputs ──────────────────────────────


ConfidenceLevel = Literal["high", "medium", "low"]


class Candidate(BaseModel):
    """A single author-affiliation extraction candidate from one specialist."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    author_name: str = Field(description="Full name as it appears in the paper")
    affiliations: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    source_specialist: str = Field(
        description="Which extractor produced this, e.g. 'header', 'footnote'"
    )
    evidence_span_ids: list[str] = Field(
        default_factory=list,
        description="IDs of ParsedDoc spans that provided the text for this candidate",
    )
    confidence: ConfidenceLevel = "medium"
    evidence_trail: EvidenceTrail = Field(default_factory=EvidenceTrail)

    @field_validator("author_name")
    @classmethod
    def _reject_trivial_names(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 2:
            raise ValueError(f"Author name too short: {stripped!r}")
        return stripped


class Candidates(BaseModel):
    """Container emitted by each specialist extractor node."""

    items: list[Candidate]
    specialist: str
    token_usage: dict[str, int] = Field(default_factory=dict)


# ─────────────────────────── planning ────────────────────────────────────────


class Plan(BaseModel):
    """Planner output — instructs the coordinator which tools to activate."""

    model_config = ConfigDict(frozen=True)

    parsers: list[str] = Field(
        description="Parser names to run, in priority order",
        examples=[["docling", "marker", "pymupdf"]],
    )
    extractors: list[str] = Field(
        description="Extractor specialist names to invoke",
        examples=[["header", "footnote", "acknowledgements", "email_domain"]],
    )
    verification_depth: Literal["light", "standard", "deep"] = "standard"
    reasons: str = Field(
        description="One-sentence explanation of the choices made"
    )


# ─────────────────────────── verdict ─────────────────────────────────────────


class Verdict(BaseModel):
    """Critic/Verifier decision on a single Candidate."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    decision: Literal["accept", "reject", "retry"]
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_band: ConfidenceLevel = "medium"
    rationale: str
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="evidence_id values from ToolEvidence items that informed this verdict",
    )
    salvage: bool = Field(
        default=False,
        description="True when the candidate was rejected by the LLM Critic but accepted "
        "via the salvage path (≥2 specialists agreed).",
    )
    retry_hint: str = Field(
        default="",
        description="Populated when decision='retry': guidance for the specialist re-run.",
    )


# ─────────────────────────── paper / source ──────────────────────────────────


class CanonicalPaper(BaseModel):
    """Normalised paper record, regardless of source."""

    model_config = ConfigDict(frozen=True)

    paper_id: str = Field(description="Source-prefixed ID, e.g. 'arxiv:2301.00001'")
    title: str
    abstract: str = ""
    venue: str | None = None
    year: int | None = None
    pdf_url: str | None = None
    source: Literal["arxiv", "openalex", "s2", "acl", "openreview", "unknown"] = "unknown"
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────── work item ───────────────────────────────────────


class WorkItem(BaseModel):
    """Top-level state object carried through the LangGraph StateGraph.

    This is the *only* object that traverses node boundaries.  Every field
    is Optional so nodes can be run independently; the coordinator validates
    that required fields are non-None before invoking downstream nodes.
    """

    # Identity
    work_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Stage outputs (filled progressively)
    canonical_paper: CanonicalPaper | None = None
    pdf_path: str | None = None
    plan: Plan | None = None

    # Parser ensemble output (stored as serialised dicts for JSON compatibility)
    parsed_doc: dict[str, Any] | None = None
    disagreement_set: dict[str, Any] | None = None

    # Extractor outputs (one per specialist)
    header_candidates: list[dict[str, Any]] = Field(default_factory=list)
    footnote_candidates: list[dict[str, Any]] = Field(default_factory=list)
    ack_candidates: list[dict[str, Any]] = Field(default_factory=list)
    email_candidates: list[dict[str, Any]] = Field(default_factory=list)

    # Merged output
    merged_candidates: list[dict[str, Any]] = Field(default_factory=list)

    # Critic verdicts
    verdicts: list[dict[str, Any]] = Field(default_factory=list)

    # Reflexion memory (per-venue verbal context injected by ReflexionStore)
    reflexion_memory: str | None = None

    # Retry routing state
    retry_count: int = 0
    critic_retry_hints: dict[str, str] = Field(
        default_factory=dict,
        description="Maps source_specialist → retry_hint for specialists that need re-run.",
    )

    # Pipeline control
    status: Literal["pending", "running", "complete", "failed"] = "pending"
    error: str | None = None
    node_timings: dict[str, float] = Field(default_factory=dict)
    token_usage: dict[str, dict[str, int]] = Field(default_factory=dict)
