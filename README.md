# Paper-Agent v2

Multi-agent system for author affiliation extraction from conference papers.
Built on LangGraph with a Planner → Specialist Extractors → Critic → Reflexion pipeline,
evaluated on **PaperAffilBench** (800 papers, 6 venues, 2021–2023).

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env          # add OPENAI_API_KEY, ANTHROPIC_API_KEY
docker-compose up --build

# v1 API:    http://localhost:8000
# v2 API:    http://localhost:8001
# UI (new):  http://localhost:3000  ← Leaderboard / Trace viewer / KG explorer
```

### Local development

```bash
pip install -r requirements.txt
cp .env.example .env

# v1 API (original pipeline + WebSocket)
uvicorn src.api.app:app --reload --port 8000

# v2 research API (leaderboard / trace / KG)
uvicorn src.v2.api.app:app --reload --port 8001

# Frontend
cd frontend && npm install && npm run dev   # http://localhost:5173
```

---

## New in v2

| Feature | Status |
|---------|--------|
| Planner → Specialist Extractor ensemble | ✅ |
| Critic with tool-grounded evidence | ✅ |
| Reflexion verbal memory per venue | ✅ |
| PDF parser ensemble (Docling, Marker, PyMuPDF) | ✅ |
| ROR / OpenAlex / S2AFF institution linkers | ✅ |
| KuzuDB knowledge graph (800 papers ingested) | ✅ |
| DHGNN-Lite collaboration forecasting | ✅ |
| PaperAffilBench 800-paper benchmark | ✅ |
| LLM-as-judge evaluation (3 judges) | ✅ |
| React frontend: Leaderboard + Trace + KG explorer | ✅ |

---

## UI Pages (new in v2)

### `/leaderboard`
System comparison table for all 7 baselines on PaperAffilBench test split.
Columns are sortable. Reads from `experiments/final/reports_cache.json`.

### `/trace/:paper_id`
Step-by-step execution trace for any paper:
1. **Plan** — which sources the Planner activated and why
2. **Parser ensemble** — Docling / Marker outputs + disagreement resolution
3. **Specialist outputs** — header / footnote / acknowledgements extractions
4. **Critic verdicts** — accept/reject/uncertain with evidence drill-down
5. **Reflexion update** — what venue memory was stored

### `/kg`
Interactive 2-hop institution collaboration subgraph (React Flow).
Select venue + year; nodes show org type (education/company).
Click a node to inspect properties. Read-only.

---

## API Reference

### v2 endpoints (`http://localhost:8001`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v2/health` | Service health check |
| `GET` | `/api/v2/leaderboard?split=test` | System comparison table |
| `GET` | `/api/v2/trace/{paper_id}` | Per-paper execution trace |
| `GET` | `/api/v2/kg/subgraph?venue=NeurIPS&year=2022&hops=2` | KG subgraph |

### v1 endpoints (`http://localhost:8000`, unchanged)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyze` | Start analysis task |
| `GET` | `/api/tasks/{id}` | Task status |
| `GET` | `/api/tasks/{id}/results` | Full results |
| `WS` | `/ws/{task_id}` | Real-time progress |

---

## Benchmark — PaperAffilBench

```
800 papers:  6 venues × 3 years × 40 + 80 hard cases
Venues:      NeurIPS, ICML, ICLR, ACL, CVPR, KDD
Years:       2021 – 2023
Splits:      train=480  dev=160  test=160
Hard cases:  multi_affiliation, name_ambiguity, consortium_author, …
```

### Leaderboard (test split, N=160)

| # | System | F1 | ROR-acc | ECE |
|---|--------|----|---------|-----|
| 1 | **Full v2** | **0.805** | 0.843 | 0.068 |
| 2 | Plan→Act+Critic | 0.781 | 0.821 | 0.082 |
| 3 | Plan→Act | 0.751 | 0.791 | 0.103 |
| 4 | OpenAlex API | 0.712 | 0.802 | 0.000 |
| 5 | v1 (frozen) | 0.693 | 0.743 | 0.124 |
| 6 | GROBID + ROR | 0.671 | 0.724 | 0.000 |
| 7 | S2AFF | 0.612 | 0.681 | 0.187 |

Full results: `experiments/final/REPORT.md` · `experiments/final/leaderboard.json`

---

## Reproduce Results

```bash
# Reproduce report from cache (<30 min, no API calls needed):
make repro

# Full run with LLM judges (requires API keys, ~$20-80):
make repro-full

# Single system:
make bench-run SYSTEM=full_v2 SPLIT=test

# Judge agreement:
make bench-agreement SPLIT=dev
```

---

## Architecture

```
                         ┌─────────────────────────────────┐
                         │         Coordinator (LangGraph)  │
                         └───────────────┬─────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
        ┌──────────┐             ┌──────────────┐           ┌──────────────┐
        │ Planner  │             │ PDF Parsers  │           │  Reflexion   │
        │(GPT-4o)  │             │ (3-way ens.) │           │  Memory      │
        └────┬─────┘             └──────┬───────┘           └──────┬───────┘
             │                          │                           │
             ▼                          ▼                           │
    ┌────────────────┐       ┌─────────────────────┐               │
    │ Source Router  │       │ Specialist Extractors│               │
    │  header        │       │  header / footnote  │               │
    │  footnote      │──────▶│  acknowledgements   │               │
    │  ack / email   │       │  email_domain / merge│               │
    └────────────────┘       └──────────┬──────────┘               │
                                        │                           │
                                        ▼                           │
                             ┌─────────────────────┐               │
                             │  Critic (tool-grounded)│◄────────────┘
                             │  ROR lookup          │
                             │  accept/reject/unc.  │
                             └──────────┬──────────┘
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │  KuzuDB KG ingest    │
                             │  + GraphRAG queries  │
                             └─────────────────────┘
```

---

## Project Layout

```
paper-agent/
├── src/
│   ├── v1/                  # v1 pipeline (frozen baseline)
│   ├── v2/
│   │   ├── api/             # FastAPI v2 endpoints
│   │   ├── agents/          # Specialist extractors + Critic
│   │   ├── orchestration/   # LangGraph coordinator + Planner + Reflexion
│   │   ├── parsers/         # Docling / Marker / PyMuPDF ensemble
│   │   ├── linkers/         # ROR / OpenAlex / S2AFF
│   │   ├── kg/              # KuzuDB schema + ingest + queries + GraphRAG
│   │   ├── eval/            # Metrics + runner + harness + budget + LLM-judge
│   │   └── analytics/       # DHGNN-Lite forecasting + baselines
│   └── api/                 # Shims → v1 (backward compat.)
├── frontend/                # React + Vite + Tailwind + React Flow
│   └── src/pages/           # LeaderboardPage, TracePage, KGPage
├── benchmark/
│   └── PaperAffilBench/     # manifest.json + gold/ + predictions_cache/
├── experiments/
│   └── final/               # REPORT.md + leaderboard.json + tables/ + figures/
├── tests/                   # 122+ tests
├── Makefile                 # repro / bench-run / kg-ingest / ...
└── docker-compose.yml       # v1 API + v2 API + frontend
```

---

## Requirements

- Python 3.11+
- Node 20+ (frontend)
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- Optional: `GOOGLE_API_KEY` (Gemini judge), `SEMANTIC_SCHOLAR_API_KEY`

## License

MIT
