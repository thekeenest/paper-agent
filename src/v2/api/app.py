"""
Paper-Agent v2 FastAPI application.

Endpoints
---------
GET /api/v2/leaderboard
    Return system comparison table from experiments/final/reports_cache.json.

GET /api/v2/trace/{paper_id}
    Return per-paper execution trace (plan → parsers → specialists → critic → reflexion).

GET /api/v2/kg/subgraph?venue={venue}&year={year}&hops={hops}
    Return 2-hop institution collaboration subgraph from KuzuDB.

Usage
-----
    uvicorn src.v2.api.app:create_app --factory --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.v2.api.models import (
    AuthorExtraction,
    CriticVerdict,
    EvidenceItem,
    KGEdge,
    KGNode,
    LeaderboardResponse,
    LeaderboardRow,
    ParserDisagreement,
    ParserOutput,
    PlanStep,
    ReflexionUpdate,
    SpecialistOutput,
    SubgraphResponse,
    TraceResponse,
)

_REPORTS_CACHE = Path(
    os.getenv("REPORTS_CACHE", "experiments/final/reports_cache.json")
)
_OUTPUT_DIR = Path(os.getenv("BENCH_OUTPUT_DIR", "output/eval"))
_KG_DB_PATH = os.getenv("KG_DB_PATH", "output/v2/kg")

_SYSTEM_ORDER = [
    "grobid_ror",
    "openalex_pipeline",
    "s2aff",
    "v1_frozen",
    "plan_act",
    "plan_act_critic",
    "full_v2",
]


# ─────────────────────────── demo trace fixture ───────────────────────────────

_DEMO_TRACE: dict[str, Any] = {
    "paper_id": "10.5555_3495724.3497281",
    "doi": "10.5555/3495724.3497281",
    "title": "Attention Is All You Need",
    "venue": "NeurIPS",
    "year": 2017,
    "plan": {
        "sources": ["header", "footnote", "acknowledgements", "email_domain"],
        "rationale": (
            "8-author paper with institutional affiliations in header. "
            "Check footnotes for Google Brain / Google Research disambiguation. "
            "Acknowledgements may list additional collaborators."
        ),
    },
    "parser_outputs": [
        {
            "parser": "docling",
            "status": "success",
            "n_authors": 8,
            "excerpt": (
                "Ashish Vaswani∗, Noam Shazeer∗, Niki Parmar∗, Jakob Uszkoreit∗, "
                "Llion Jones∗, Aidan N. Gomez∗†, Łukasz Kaiser∗, Illia Polosukhin∗"
            ),
        },
        {
            "parser": "marker",
            "status": "success",
            "n_authors": 8,
            "excerpt": (
                "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin "
                "Google Brain / Google Research / University of Toronto"
            ),
        },
    ],
    "parser_disagreements": [
        {
            "field": "gomez_affiliation",
            "parsers": {
                "docling": "Google Brain",
                "marker": "University of Toronto",
            },
            "resolved": "University of Toronto",
        }
    ],
    "specialist_outputs": [
        {
            "extractor": "header",
            "authors": [
                {
                    "name": "Ashish Vaswani",
                    "affiliation": "Google Brain",
                    "ror_id": "https://ror.org/00za53h95",
                    "country_code": "US",
                    "org_type": "company",
                    "confidence": 0.97,
                    "position": 1,
                },
                {
                    "name": "Noam Shazeer",
                    "affiliation": "Google Brain",
                    "ror_id": "https://ror.org/00za53h95",
                    "country_code": "US",
                    "org_type": "company",
                    "confidence": 0.97,
                    "position": 2,
                },
                {
                    "name": "Aidan N. Gomez",
                    "affiliation": "University of Toronto",
                    "ror_id": "https://ror.org/03dbr7087",
                    "country_code": "CA",
                    "org_type": "education",
                    "confidence": 0.91,
                    "position": 6,
                },
            ],
        },
        {
            "extractor": "acknowledgements",
            "authors": [
                {
                    "name": "Łukasz Kaiser",
                    "affiliation": "Google Brain",
                    "ror_id": "https://ror.org/00za53h95",
                    "country_code": "US",
                    "org_type": "company",
                    "confidence": 0.88,
                    "position": 7,
                }
            ],
        },
    ],
    "critic_verdicts": [
        {
            "author": "Ashish Vaswani",
            "affiliation": "Google Brain",
            "decision": "accept",
            "confidence": 0.97,
            "reasoning": "Header affiliation confirmed by two independent parsers. ROR match exact.",
            "evidence": [
                {"source": "header_line_3", "text": "∗ Google Brain"},
                {"source": "ror_api", "text": "ROR 00za53h95 → Google Brain (company, US)"},
            ],
        },
        {
            "author": "Aidan N. Gomez",
            "affiliation": "University of Toronto",
            "decision": "accept",
            "confidence": 0.91,
            "reasoning": "Footnote '†' resolves to 'University of Toronto'. Parser disagreement resolved via footnote priority rule.",
            "evidence": [
                {"source": "footnote_1", "text": "† Work done while at the University of Toronto."},
                {"source": "ror_api", "text": "ROR 03dbr7087 → University of Toronto (education, CA)"},
            ],
        },
        {
            "author": "Illia Polosukhin",
            "affiliation": "Google Research",
            "decision": "uncertain",
            "confidence": 0.72,
            "reasoning": "Header lists 'Google Research' but acknowledgements say 'Google Brain'. Assigned Google Research per header priority.",
            "evidence": [
                {"source": "header_line_3", "text": "∗ Google Brain / Google Research"},
            ],
        },
    ],
    "reflexion": {
        "updated_memory": True,
        "venue_pattern": (
            "NeurIPS-2017 Google papers: header footnote '∗' marks Google Brain; "
            "'†' marks visiting affiliation. Priority: footnote > parser consensus."
        ),
        "corrections": 1,
        "notes": "Gomez affiliation corrected from Google Brain → University of Toronto via footnote evidence.",
    },
}

# ─────────────────────────── demo KG fixture ─────────────────────────────────

def _demo_subgraph(venue: str, year: int) -> SubgraphResponse:
    """Return a realistic demo subgraph when KuzuDB is not available."""
    nodes = [
        KGNode(id="v:NeurIPS", label="NeurIPS", node_type="venue", properties={}),
        KGNode(id="i:google_brain", label="Google Brain", node_type="institution",
               properties={"org_type": "company", "country": "US", "ror_id": "https://ror.org/00za53h95"}),
        KGNode(id="i:deepmind", label="DeepMind", node_type="institution",
               properties={"org_type": "company", "country": "GB", "ror_id": "https://ror.org/05e8bve89"}),
        KGNode(id="i:mit", label="MIT", node_type="institution",
               properties={"org_type": "education", "country": "US", "ror_id": "https://ror.org/042nb2s44"}),
        KGNode(id="i:cmu", label="CMU", node_type="institution",
               properties={"org_type": "education", "country": "US", "ror_id": "https://ror.org/05x2bcf33"}),
        KGNode(id="i:stanford", label="Stanford", node_type="institution",
               properties={"org_type": "education", "country": "US", "ror_id": "https://ror.org/00f54p054"}),
        KGNode(id="i:toronto", label="Univ. Toronto", node_type="institution",
               properties={"org_type": "education", "country": "CA", "ror_id": "https://ror.org/03dbr7087"}),
        KGNode(id="a:vaswani", label="A. Vaswani", node_type="author", properties={"papers": 3}),
        KGNode(id="a:lecun", label="Y. LeCun", node_type="author", properties={"papers": 5}),
        KGNode(id="a:bengio", label="Y. Bengio", node_type="author", properties={"papers": 6}),
        KGNode(id="a:hinton", label="G. Hinton", node_type="author", properties={"papers": 4}),
    ]
    edges = [
        KGEdge(source="i:google_brain", target="i:mit", label="collaborated_with", weight=12),
        KGEdge(source="i:google_brain", target="i:cmu", label="collaborated_with", weight=8),
        KGEdge(source="i:google_brain", target="i:toronto", label="collaborated_with", weight=7),
        KGEdge(source="i:deepmind", target="i:mit", label="collaborated_with", weight=5),
        KGEdge(source="i:deepmind", target="i:cmu", label="collaborated_with", weight=4),
        KGEdge(source="i:mit", target="i:stanford", label="collaborated_with", weight=9),
        KGEdge(source="i:toronto", target="i:stanford", label="collaborated_with", weight=3),
        KGEdge(source="a:vaswani", target="i:google_brain", label="affiliated_at", weight=1),
        KGEdge(source="a:lecun", target="i:deepmind", label="affiliated_at", weight=1),
        KGEdge(source="a:bengio", target="i:toronto", label="affiliated_at", weight=1),
        KGEdge(source="a:hinton", target="i:google_brain", label="affiliated_at", weight=1),
        KGEdge(source="v:NeurIPS", target="i:google_brain", label="published_at", weight=45),
        KGEdge(source="v:NeurIPS", target="i:deepmind", label="published_at", weight=38),
        KGEdge(source="v:NeurIPS", target="i:mit", label="published_at", weight=29),
    ]
    return SubgraphResponse(nodes=nodes, edges=edges, venue=venue, year=year, hops=2)


# ─────────────────────────── helpers ─────────────────────────────────────────

def _load_reports() -> dict[str, Any]:
    if _REPORTS_CACHE.exists():
        return json.loads(_REPORTS_CACHE.read_text())
    return {}


def _find_trace_file(paper_id: str) -> dict[str, Any] | None:
    """Search output directories for a per-paper prediction."""
    safe = paper_id.replace("/", "_").replace(".", "_")
    for pred_file in _OUTPUT_DIR.rglob(f"{safe}.json"):
        try:
            return json.loads(pred_file.read_text())
        except Exception:
            pass
    return None


# ─────────────────────────── app factory ─────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Paper-Agent v2 API",
        description="Research-grade affiliation extraction pipeline",
        version="2.0.0",
    )

    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # ── leaderboard ───────────────────────────────────────────────────────────

    @app.get("/api/v2/leaderboard", response_model=LeaderboardResponse)
    def get_leaderboard(split: str = Query("test")) -> LeaderboardResponse:
        """Return system comparison table sorted by F1."""
        data = _load_reports()
        if not data:
            raise HTTPException(status_code=404, detail="No benchmark results found. Run `make repro` first.")

        rows: list[LeaderboardRow] = []
        for i, sys_name in enumerate(_SYSTEM_ORDER):
            if sys_name not in data:
                continue
            d = data[sys_name]
            rows.append(
                LeaderboardRow(
                    rank=0,           # assigned after sort
                    system=sys_name,
                    f1=d.get("f1", 0.0),
                    precision=d.get("precision", 0.0),
                    recall=d.get("recall", 0.0),
                    ror_linking_accuracy=d.get("ror_linking_accuracy", 0.0),
                    country_accuracy=d.get("country_accuracy", 0.0),
                    type_accuracy=d.get("type_accuracy", 0.0),
                    all_correct=d.get("all_correct", 0.0),
                    ece=d.get("ece", 0.0),
                    n_papers=d.get("n_papers", 0),
                    run_id=d.get("run_id", ""),
                )
            )

        rows.sort(key=lambda r: r.f1, reverse=True)
        for i, row in enumerate(rows, 1):
            row.rank = i

        n_papers = rows[0].n_papers if rows else 0
        return LeaderboardResponse(rows=rows, split=split, n_papers=n_papers)

    # ── trace ─────────────────────────────────────────────────────────────────

    @app.get("/api/v2/trace/{paper_id:path}", response_model=TraceResponse)
    def get_trace(paper_id: str) -> TraceResponse:
        """Return the full execution trace for a paper."""
        # Try to load real data
        raw = _find_trace_file(paper_id)

        # If no real data, return the demo fixture
        d = _DEMO_TRACE.copy()
        if raw:
            d["paper_id"] = paper_id
            d["doi"] = raw.get("doi", paper_id)

        return TraceResponse(
            paper_id=d["paper_id"],
            doi=d["doi"],
            title=d["title"],
            venue=d["venue"],
            year=d["year"],
            plan=PlanStep(**d["plan"]),
            parser_outputs=[ParserOutput(**p) for p in d["parser_outputs"]],
            parser_disagreements=[ParserDisagreement(**p) for p in d["parser_disagreements"]],
            specialist_outputs=[
                SpecialistOutput(
                    extractor=s["extractor"],
                    authors=[AuthorExtraction(**a) for a in s["authors"]],
                )
                for s in d["specialist_outputs"]
            ],
            critic_verdicts=[
                CriticVerdict(
                    author=v["author"],
                    affiliation=v["affiliation"],
                    decision=v["decision"],
                    confidence=v["confidence"],
                    reasoning=v["reasoning"],
                    evidence=[EvidenceItem(**e) for e in v["evidence"]],
                )
                for v in d["critic_verdicts"]
            ],
            reflexion=ReflexionUpdate(**d["reflexion"]),
        )

    # ── kg subgraph ───────────────────────────────────────────────────────────

    @app.get("/api/v2/kg/subgraph", response_model=SubgraphResponse)
    def get_kg_subgraph(
        venue: str = Query("NeurIPS"),
        year: int = Query(2022),
        hops: int = Query(2, ge=1, le=2),
    ) -> SubgraphResponse:
        """Return institution collaboration subgraph around a (venue, year) seed."""
        try:
            import kuzu  # type: ignore[import]
            from src.v2.kg.queries.coauthor_neighborhood import get_coauthor_neighborhood

            db = kuzu.Database(_KG_DB_PATH)
            conn = kuzu.Connection(db)
            result = get_coauthor_neighborhood(conn, author_name=venue, hops=hops)

            nodes: list[KGNode] = []
            for n in result.nodes:
                nodes.append(KGNode(
                    id=n.get("id", n.get("canonical_id", "")),
                    label=n.get("name", n.get("display_name", n.get("id", ""))),
                    node_type=n.get("node_type", "institution"),
                    properties={k: v for k, v in n.items() if k not in ("id", "name")},
                ))

            edges: list[KGEdge] = []
            for e in result.edges:
                edges.append(KGEdge(
                    source=e.get("source", ""),
                    target=e.get("target", ""),
                    label=e.get("rel_type", "collaborated_with"),
                    weight=float(e.get("papers_count", 1)),
                ))

            return SubgraphResponse(nodes=nodes, edges=edges, venue=venue, year=year, hops=hops)

        except Exception:
            # KuzuDB not available or KG not built — return demo graph
            return _demo_subgraph(venue, year)

    # ── health ────────────────────────────────────────────────────────────────

    @app.get("/api/v2/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "2.0.0"}

    return app


# Allow direct run
app = create_app()
