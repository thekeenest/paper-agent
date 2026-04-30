# Paper-Agent v2 — Architecture Overview

> **This document is a navigational stub.**
> The authoritative design lives in [`coursework_v2/DEV_PLAN.md`](../coursework_v2/DEV_PLAN.md) (§3 "System Architecture").
> Read that first; this file only maps DEV_PLAN concepts to repository paths.

---

## Repository layout

```
paper-agent/
├── src/
│   ├── v1/                  # Frozen Spring-2025 pipeline (preserved verbatim)
│   │   ├── graph.py         # LangGraph linear pipeline: search→download→parse→extract→normalize→aggregate
│   │   ├── nodes.py         # Six LangGraph node functions
│   │   ├── models.py        # Pydantic data models (PaperMetadata, AuthorAffiliation, …)
│   │   ├── normalizer.py    # Local KB (~50 orgs) + fuzzy + LLM fallback
│   │   ├── analytics.py     # CSV → value_counts() → matplotlib
│   │   ├── evaluation.py    # Gold-standard F1 evaluation
│   │   ├── data_sources/    # ArXiv, S2, OpenAlex, ROR clients
│   │   └── api/             # FastAPI + WebSocket backend
│   │
│   ├── v2/                  # New development — import from src.v2.*
│   │   ├── orchestration/   # Coordinator, Planner, ReflexionController  [DEV_PLAN §3.2]
│   │   ├── agents/
│   │   │   ├── extractors/  # Header, Footnote, EmailDomain, Acknowledgements [DEV_PLAN §3.3]
│   │   │   └── critic/      # Grounded Critic/Verifier with tool-cited evidence [DEV_PLAN §3.4]
│   │   ├── parsers/         # Docling + Marker + Nougat + PyMuPDF ensemble [DEV_PLAN §3.2]
│   │   ├── linkers/         # ROR, OpenAlex, S2AND, DNS resolvers [DEV_PLAN §3.3]
│   │   ├── kg/              # KuzuDB schema + ingest + GraphRAG queries [DEV_PLAN §3.5]
│   │   ├── analytics/       # Trend analytics, country heatmaps, GNN forecasting [DEV_PLAN §3.6]
│   │   └── eval/            # PaperAffilBench harness + baselines [DEV_PLAN §4]
│   │
│   ├── analytics/           # Re-export shim (v1 AnalyticsEngine) + future v2 analytics
│   └── *.py                 # Re-export shims keeping legacy src.* imports alive
│
├── benchmark/
│   └── PaperAffilBench/     # ~800 manually verified papers (annotation sprint: Week 10)
│       ├── papers/          # Raw PDFs (gitignored)
│       ├── gold/            # Per-paper annotation JSON
│       └── splits/          # train/dev/test manifests
│
├── experiments/             # Reproducible experiment artefacts (configs + results)
├── tests/                   # Pytest suite — structural + unit + integration
└── coursework_v2/           # Research planning documents
    ├── DEV_PLAN.md          # Gap analysis, architecture, milestones, risk register
    ├── READING_LIST.md      # 35-entry annotated bibliography
    └── COURSEWORK_OUTLINE.md # Section-by-section .tex skeleton
```

---

## v2 Component summary

Each row maps to one sentence from [DEV_PLAN.md §3](../coursework_v2/DEV_PLAN.md).

| Component | Path | One-line purpose |
|---|---|---|
| **Coordinator** | `src/v2/orchestration/coordinator.py` | Root LangGraph StateGraph; dispatches to parser and extractor sub-graphs |
| **Planner** | `src/v2/orchestration/planner.py` | Decides parser order, verification depth, and source priority per paper |
| **ReflexionController** | `src/v2/orchestration/reflexion.py` | Maintains per-venue verbal memory; updates extraction policy after failures |
| **Parser Ensemble** | `src/v2/parsers/ensemble.py` | Runs Docling → Marker → Nougat → PyMuPDF; surfaces author-block disagreement |
| **HeaderExtractor** | `src/v2/agents/extractors/header.py` | Extracts author block from PDF header region |
| **FootnoteExtractor** | `src/v2/agents/extractors/footnote.py` | Parses affiliation footnotes with superscript markers |
| **EmailDomainExtractor** | `src/v2/agents/extractors/email_domain.py` | Infers institution from author email domains |
| **AcknowledgementsAgent** | `src/v2/agents/extractors/acknowledgements.py` | Extracts secondary affiliations from Acknowledgements section |
| **Critic/Verifier** | `src/v2/agents/critic/critic.py` | Grades extractor candidates with cited tool evidence (OpenAlex / ROR / DNS / S2) |
| **RORLinker** | `src/v2/linkers/ror_linker.py` | Organization string → ROR ID + canonical name + country |
| **OpenAlexLinker** | `src/v2/linkers/openalex_linker.py` | Paper ID → author affiliations + institution IDs from OpenAlex |
| **S2ANDLinker** | `src/v2/linkers/s2and_linker.py` | Author name + context → disambiguated S2 author ID |
| **KG Schema** | `src/v2/kg/schema.py` | KuzuDB DDL — Author / Paper / Institution / Venue / Topic / Evidence nodes |
| **KG Ingest** | `src/v2/kg/ingest.py` | Idempotent upsert of extraction results into the knowledge graph |
| **GraphRAG** | `src/v2/kg/graphrag.py` | Natural-language Q&A over the KG via LangChain |
| **PaperAffilBench** | `src/v2/eval/harness.py` | Runs any extractor on the ~800-paper benchmark; computes F1 / ROR-acc / ECE |

---

## Data flow (v2 target)

```
[Paper ID / Query]
        │
        ▼
  [Planner]  ──selects── source priority, parser order, verification depth
        │
        ▼
  [Source Router]  ──fetches── PDF + metadata from ArXiv / OpenAlex / S2 / ACL
        │
        ▼
  [Parser Ensemble]  ──Docling▸Marker▸Nougat▸PyMuPDF── disagreement score
        │
        ▼
  [Specialist Extractors]  ──Header / Footnote / Email / Ack──  ExtractionCandidates[]
        │
        ▼
  [Critic / Verifier]  ──OpenAlex / ROR / DNS / S2──  EvidenceRecord + verdict
        │
        ├── accept → [KG Ingest] → KuzuDB
        └── reject → [ReflexionController] → update policy → retry or escalate
```

---

## Key design decisions (see DEV_PLAN.md §3 for rationale)

- **KuzuDB** preferred over Neo4j — embedded, no Docker dependency, Cypher-compatible.
- **Docling** is primary parser — open-source, layout-aware, better two-column recall.
- **Claude Sonnet 4.6** as Critic LLM — separate role from extractor to reduce sycophancy.
- **Pydantic ≥ 2** throughout v2 — strict mode, no arbitrary types in new code.
- **Python 3.11** minimum — match-statement, `tomllib`, better error messages.
