# Paper-Agent v2 — Engineering Recap (Stages 0–11)

This document summarises everything built in v2, what changed relative to v1,
the research papers that informed the design, and the complete new architecture.

---

## Stages Completed

| Stage | Name | Key Deliverables |
|-------|------|-----------------|
| 0 | Project bootstrap | pyproject.toml, ruff, mypy, pytest infrastructure |
| 1 | PDF parser ensemble | Docling (primary), Marker, Nougat, PyMuPDF; schema.py |
| 2 | Institution linkers | ROR fuzzy linker, OpenAlex institutions, S2AFF, S2AND dedup |
| 3 | Orchestration | LangGraph coordinator, contracts (WorkItem/CandidateRecord), source router |
| 4 | Planner | GPT-4o planner selects source set per paper |
| 5 | Specialist extractors | header, footnote, acknowledgements, email_domain, merge |
| 6 | Critic | Tool-grounded verifier: accept/reject/uncertain + salvage path |
| 7 | Reflexion | Verbal memory store per venue; reflector loop on low-confidence papers |
| 8 | KG layer | KuzuDB 0.11.3 schema + ingest + 4 Cypher queries + GraphRAG |
| 9 | Benchmark | PaperAffilBench v2.0 — 800 papers, manifest, annotate CLI, metrics, runner |
| 10 | Analytics & forecasting | DHGNN-Lite + StaticGCN + co_freq baselines; forecasting notebook |
| 11 | Stage-8 harness + UI | Full eval harness, LLM-as-judge, leaderboard, React frontend |

---

## What Changed vs v1

### v1 architecture
```
Search → Fetch PDF → PyMuPDF parse → LLM extract (single call) → Normalise (KB fuzzy) → Aggregate
```
- Single monolithic LLM extraction call
- Rule-based fuzzy matching against a hardcoded knowledge base
- No verification step
- No per-venue learning
- No structured benchmark

### v2 architecture
```
Planner → 3-way PDF ensemble → Specialist Extractors (4 types) → Critic (tool-grounded)
       → Reflexion memory                                       → KuzuDB KG ingest
```

**Key differences:**

| Dimension | v1 | v2 |
|-----------|----|----|
| PDF parsing | PyMuPDF only | Docling + Marker + PyMuPDF ensemble; disagreement resolution |
| Extraction | Single LLM call | 4 specialist extractors per source type |
| Verification | None | Critic with ROR/OpenAlex tool calls |
| Memory | None | Reflexion verbal memory, persisted per venue |
| Institution linking | Hardcoded KB + fuzzy | ROR API + OpenAlex + S2AFF, all with async caching |
| Benchmarking | 100 papers, internal | PaperAffilBench 800 papers, 7 systems, 3 LLM judges |
| Knowledge graph | None | KuzuDB (Author/Institution/Paper/Venue nodes; collaboration edges) |
| Forecasting | None | DHGNN-Lite temporal GNN for institution link prediction |
| Frontend | Task dashboard only | + Leaderboard + Trace viewer + KG explorer |

---

## Research Papers Applied

### PDF Parsing
- **Docling** (Auer et al., 2024) — IBM deep-learning PDF parser, primary backbone
- **Marker** (Nougat successor) — transformer OCR for difficult PDFs
- **Nougat** (Blecher et al., NeurIPS 2023) — academic PDF→markup via vision transformer

### Institution Linking
- **ROR** (Lammey, 2020) — Research Organization Registry; used for canonical IDs
- **OpenAlex** (Priem et al., 2022) — open bibliographic database with institution graph
- **S2AFF** (Lo et al., 2023, Semantic Scholar) — ML-based affiliation disambiguation
- **S2AND** (Subramanian et al., 2021) — author name disambiguation via clustering

### Orchestration / LLM
- **LangGraph** (Chase et al., 2024) — stateful multi-agent graph execution
- **ReAct** (Yao et al., ICLR 2023) — reasoning + acting pattern for Critic tool use
- **Reflexion** (Shinn et al., NeurIPS 2023) — verbal memory for self-correction across runs

### Knowledge Graph
- **KuzuDB** (Jin et al., 2023) — embeddable columnar graph database; MERGE-based ingest
- **GraphRAG** (Edge et al., 2024) — entity-linked subgraph retrieval for LLM grounding

### Forecasting
- **DHGNN** (based on HAN, Wang et al., WWW 2019) — heterogeneous attention network;
  adapted as DHGNN-Lite (≤5M params) with GRU temporal state
- **GAE** (Kipf & Welling, ICLR 2016) — graph autoencoder pattern for link prediction
- **Static GCN** (Kipf & Welling, 2017) — mean-aggregation GCN baseline

### Evaluation
- **BERTScore** (Zhang et al., 2020) — surface-form affiliation similarity
- **LLM-as-judge** (Zheng et al., 2023) — cross-model evaluation protocol
- **Cohen's κ** (Cohen, 1960) — inter-rater agreement
- **Krippendorff's α** (Krippendorff, 2011) — multi-rater ordinal agreement
- **ECE** (Naeini et al., 2015) — expected calibration error for confidence evaluation

---

## New Architecture Detail

### Coordinator graph (LangGraph)

```
START
  └→ planner_node          # GPT-4o plans which sources to activate
       └→ source_router     # dispatches to PDF parsers
            └→ parser_ensemble  # Docling / Marker / PyMuPDF → disagreement merge
                 └→ specialist_extractors  # header / footnote / ack / email
                      └→ linker_node       # ROR + OpenAlex + S2AFF
                           └→ critic_node  # tool-grounded accept/reject/uncertain
                                └→ reflexion_node  # verbal memory update
                                     └→ kg_ingest   # KuzuDB MERGE upsert
                                          └→ END
```

### PDF Parser Ensemble (`src/v2/parsers/`)
- `docling_parser.py` — primary; structured JSON with block-level layout
- `marker_parser.py` — fallback; good on scanned PDFs
- `pymupdf_parser.py` — fast; used for header/footnote extraction
- `ensemble.py` — majority-vote disagreement resolution per field

### Specialist Extractors (`src/v2/agents/extractors/`)
- `header.py` — author list + superscript affiliation markers
- `footnote.py` — resolves footnote symbols to institution names
- `acknowledgements.py` — extracts gratitude-section affiliations
- `email_domain.py` — maps email domains to ROR institutions
- `merge.py` — combines all extractors with confidence weighting

### Critic (`src/v2/agents/critic/`)
- `critic.py` — GPT-4o with tool-use: accepts, rejects, or flags uncertain
- `tool_evidence.py` — calls ROR API, OpenAlex, email-domain lookup
- Returns `CriticVerdict` with `decision`, `confidence`, `evidence_trail`
- Uncertain verdicts go to a salvage path (re-extraction with different source)

### Reflexion (`src/v2/orchestration/reflexion.py`)
- Per-venue verbal memory: e.g. "NeurIPS-2022 Google papers use footnote '∗' for Google Brain"
- Persisted as JSON in `output/v2/reflexion/`
- Applied at plan time to pre-activate high-signal sources

### KG Layer (`src/v2/kg/`)
- **KuzuDB** embedded graph: Author, Institution, Paper, Venue nodes
- **MERGE-based ingest**: idempotent; 100 work items in ~40s
- **Cypher queries**: industry_share_by_venue, country_strategy_heatmap,
  coauthor_neighborhood (2-hop), top_institutions_by_venue
- **GraphRAG**: entity-linked subgraph → LLM synthesis for natural-language answers

### DHGNN-Lite (`src/v2/analytics/forecasting/dhgnn_lite.py`)
- Heterogeneous Attention Network layers (Author + Institution node types)
- GRU temporal state across yearly snapshots
- ≤5M parameters (enforced by assertion)
- Trained on 2019–2022, evaluated on 2023
- Task: predict which institution pairs will co-author in the next year
- Constraint: DHGNN-Lite AUC ≥ co_freq AUC + 0.05

### Benchmark — PaperAffilBench (`benchmark/PaperAffilBench/`)
```
800 papers  ·  6 venues  ·  3 years (2021–2023)
80 hard cases tagged with: multi_affiliation, name_ambiguity,
  non_english_affiliation, missing_ror, industry_academia_joint,
  consortium_author, position_ambiguity, historical_institution
Splits: train=480  dev=160  test=160
Gold schema: {paper_id, doi, title, year, venue, annotator,
              annotations: [{author_name, normalized_aff, ror_id,
                             country_code, type, evidence_span, email}]}
```

### Evaluation Harness (`src/v2/eval/`)
- `metrics.py` — P/R/F1, ROR-acc, country-acc, type-acc, all_correct, ECE
- `runner.py` — deterministic run_id (SHA-256), 7-system evaluation
- `budget.py` — thread-safe BudgetTracker; MAX_USD hard ceiling
- `llm_judge.py` — 3-judge protocol (Claude, GPT-4o, Gemini); cached per DOI
- `agreement.py` — Cohen's κ, Krippendorff's α; pairwise + macro-average
- `harness.py` — full Stage-8 orchestrator; Tables 4.2–4.4; Figures 4.2–4.6;
  H1–H4 hypothesis tests with 95% CIs; REPORT.md + leaderboard.md

### Frontend (`frontend/`)
- Vite + React 18 + TypeScript
- Tailwind CSS for styling
- React Flow (`@xyflow/react`) for KG subgraph visualization
- React Router v6 for SPA routing
- **Pages**: `/leaderboard` (sortable table), `/trace/:paper_id` (5-step trace), `/kg` (2-hop subgraph)
- **API client** (`src/api.ts`) proxies to `src/v2/api/app.py` at `localhost:8001`

---

## Test Coverage

```
tests/test_kg.py          32 tests  (KuzuDB schema, ingest, queries, GraphRAG)
tests/test_orchestration/ ~40 tests  (coordinator, planner, critic, reflexion)
tests/test_eval/          ~20 tests  (metrics, runner, budget)
tests/test_linkers/       ~10 tests  (ROR, OpenAlex, S2AND)
tests/test_parsers/       ~10 tests  (Docling, ensemble disagreement)
tests/test_forecasting/   ~10 tests  (dataset, DHGNN, co_freq)
─────────────────────────────────────────
Total:                   ~122 tests
```

---

## Make Targets

```bash
make repro              # Reproduce REPORT.md from cache (<30 min)
make repro-full         # Full harness run (requires API keys)
make bench-run SYSTEM=full_v2 SPLIT=test
make bench-agreement    # Cohen κ + Krippendorff α
make kg-ingest INPUT=...
make kg-query Q="industry_share_by_venue --venue NeurIPS"
make test               # Full test suite
make leaderboard        # Regenerate benchmark/leaderboard.md
```

---

## File Count Summary

| Layer | Files added in v2 |
|-------|--------------------|
| Orchestration | 6 |
| Parsers | 6 |
| Agents (extractors + critic) | 8 |
| Linkers | 6 |
| KG (schema + queries + graphrag) | 8 |
| Eval (metrics + harness + judge) | 8 |
| Analytics (forecasting) | 6 |
| Frontend (React pages + components) | 12 |
| API v2 | 3 |
| Tests | 5 test files |
| **Total new files** | **~68** |
