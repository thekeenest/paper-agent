"""Pydantic response models for the v2 API."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel


# ─────────────────────────── leaderboard ─────────────────────────────────────

class LeaderboardRow(BaseModel):
    rank: int
    system: str
    f1: float
    precision: float
    recall: float
    ror_linking_accuracy: float
    country_accuracy: float
    type_accuracy: float
    all_correct: float
    ece: float
    n_papers: int
    run_id: str


class LeaderboardResponse(BaseModel):
    rows: list[LeaderboardRow]
    split: str
    n_papers: int


# ─────────────────────────── trace ───────────────────────────────────────────

class PlanStep(BaseModel):
    sources: list[str]
    rationale: str


class ParserOutput(BaseModel):
    parser: str
    status: str
    n_authors: int
    excerpt: str


class ParserDisagreement(BaseModel):
    field: str
    parsers: dict[str, str]
    resolved: str


class AuthorExtraction(BaseModel):
    name: str
    affiliation: str
    ror_id: str
    country_code: str
    org_type: str
    confidence: float
    position: int


class SpecialistOutput(BaseModel):
    extractor: str
    authors: list[AuthorExtraction]


class EvidenceItem(BaseModel):
    source: str
    text: str


class CriticVerdict(BaseModel):
    author: str
    affiliation: str
    decision: str          # accept | reject | uncertain
    confidence: float
    reasoning: str
    evidence: list[EvidenceItem]


class ReflexionUpdate(BaseModel):
    updated_memory: bool
    venue_pattern: str
    corrections: int
    notes: str


class TraceResponse(BaseModel):
    paper_id: str
    doi: str
    title: str
    venue: str
    year: int
    plan: PlanStep
    parser_outputs: list[ParserOutput]
    parser_disagreements: list[ParserDisagreement]
    specialist_outputs: list[SpecialistOutput]
    critic_verdicts: list[CriticVerdict]
    reflexion: ReflexionUpdate


# ─────────────────────────── KG subgraph ─────────────────────────────────────

class KGNode(BaseModel):
    id: str
    label: str
    node_type: str          # author | institution | paper | venue
    properties: dict[str, Any] = {}


class KGEdge(BaseModel):
    source: str
    target: str
    label: str
    weight: float = 1.0


class SubgraphResponse(BaseModel):
    nodes: list[KGNode]
    edges: list[KGEdge]
    venue: str
    year: int
    hops: int
