# Message to Professor — Paper-Agent v2 Update

---

Dear Professor,

Since submitting v1 of Paper-Agent I have substantially redesigned and extended
the system. Below is a concise summary of everything new in v2, why each piece
was built, and the research it is grounded in.

---

## What Was v1

v1 was a linear multi-agent pipeline that:
- Searched ArXiv / Semantic Scholar / OpenAlex for papers
- Downloaded PDFs and parsed them with PyMuPDF
- Made a single GPT-4o-mini call to extract author affiliations
- Normalised organisation names against a hand-crafted knowledge base using fuzzy string matching
- Aggregated results and produced CSV/JSON/plots

It achieved ~80-85% affiliation extraction accuracy on a small internal test set.
The main limitations were: no verification step, no learning across papers,
no systematic benchmark, and brittle handling of complex affiliation formats
(multi-author footnotes, consortium authors, non-English institutions).

---

## What Is New in v2

### 1. Planner → Specialist Extractors → Critic → Reflexion pipeline

Instead of one LLM call, v2 uses a **four-stage LangGraph graph**:

1. **Planner** — a GPT-4o reasoning step that reads the paper metadata and decides
   which extraction sources to activate (header, footnote, acknowledgements, email domains).
   This mirrors the ReAct pattern (Yao et al., ICLR 2023).

2. **Specialist Extractors** — four dedicated agents, each optimised for one section
   of a paper. The header extractor handles superscript affiliation markers (e.g. ∗, †, 1, 2).
   The footnote extractor resolves those markers. The acknowledgements extractor
   finds secondary affiliations. The email-domain extractor maps email addresses
   to ROR institution IDs. Their outputs are merged with confidence weighting.

3. **Critic** — a tool-grounded verifier (Critic pattern, similar to Constitutional AI).
   It calls the ROR API and OpenAlex API as tools to verify each extracted affiliation,
   then returns `accept`, `reject`, or `uncertain` with an evidence trail.
   Uncertain predictions go to a salvage path that re-extracts from a different source.

4. **Reflexion memory** (Shinn et al., NeurIPS 2023) — after processing each paper,
   the system writes a natural-language observation about the venue's affiliation patterns
   into a persistent per-venue memory store. On subsequent papers from the same venue,
   the Planner reads this memory and pre-activates the most effective sources.
   This is essentially few-shot in-context learning accumulated over the benchmark run.

### 2. PDF Parser Ensemble

v1 used only PyMuPDF. v2 runs three parsers in parallel:
- **Docling** (IBM, 2024) — deep-learning layout parser with block-level structure
- **Marker** — transformer-based OCR, good on scanned PDFs
- **PyMuPDF** — fast, used as the baseline

A disagreement-resolution step takes the majority consensus per extracted field.
This reduced extraction errors on papers with complex multi-column layouts by ~15%.

### 3. Institution Linkers

v1 used a hardcoded knowledge base (~2,000 entries). v2 has three live linkers:
- **ROR** (Research Organization Registry) — canonical institution IDs via fuzzy API lookup
- **OpenAlex** — open bibliographic database with 100M+ works; used for institution graph
- **S2AFF / S2AND** (Allen Institute, 2021–2023) — ML-based disambiguation and
  author-name deduplication trained on Semantic Scholar data

All linkers use an async SQLite cache to avoid redundant API calls.

### 4. PaperAffilBench — a new benchmark

v1 had no public benchmark. I created **PaperAffilBench v2.0**:
- **800 papers** from NeurIPS, ICML, ICLR, ACL, CVPR, KDD (2021–2023)
- **80 hard cases** tagged by type: multi_affiliation, name_ambiguity,
  non_english_affiliation, missing_ror, industry_academia_joint,
  consortium_author, position_ambiguity, historical_institution
- Train / dev / test splits (480 / 160 / 160)
- Gold annotations follow the ROR schema: `{author_name, normalized_aff, ror_id, country_code, org_type}`

Seven systems are evaluated: the new full v2 pipeline, two ablations (without Critic;
without Reflexion), and four baselines (GROBID+ROR, OpenAlex API, S2AFF, v1 frozen).

**Test-split results (N=160 papers):**

| System | F1 | ROR-acc |
|--------|----|---------|
| Full v2 | **0.805** | 0.843 |
| + Critic (no Reflexion) | 0.781 | 0.821 |
| Plan→Act only | 0.751 | 0.791 |
| OpenAlex API | 0.712 | 0.802 |
| v1 (frozen) | 0.693 | 0.743 |
| GROBID + ROR | 0.671 | 0.724 |
| S2AFF | 0.612 | 0.681 |

The ablation confirms that both the Critic (+3.0 F1 points) and Reflexion memory
(+2.4 F1 points) add measurable value.

### 5. LLM-as-Judge Evaluation

To assess output quality beyond exact-match F1, I implemented a three-judge
cross-model evaluation protocol following Zheng et al. (2023):
- **Claude Sonnet 4.6**, **GPT-4o**, **Gemini 1.5 Flash** each score every
  extracted affiliation on a 0/1/2 scale (wrong/partial/correct)
- Results are cached per (doi, judge, system)
- Inter-judge agreement is measured with **Cohen's κ** and **Krippendorff's α**
- A budget guard (`MAX_USD` env var, default $80) hard-caps API spend

### 6. KuzuDB Knowledge Graph

All extracted affiliations are ingested into a **KuzuDB** embedded columnar graph
(Author / Institution / Paper / Venue nodes; COAUTHORED_WITH / AFFILIATED_AT /
COLLABORATED_WITH edges). This enables four analytical queries:
- Industry share by venue and year
- Country collaboration heatmap
- 2-hop co-author neighbourhood
- Top institutions by venue

A **GraphRAG** module does entity-linked subgraph retrieval and answers
natural-language questions about collaboration patterns.

### 7. Collaboration Forecasting (DHGNN-Lite)

I implemented a small heterogeneous temporal graph neural network for the task
"predict which institution pairs will co-author at venue V in year Y+1":

- **DHGNN-Lite** — Heterogeneous Attention Network (Wang et al., WWW 2019)
  with GRU temporal state across yearly snapshots, ≤5M parameters
- **Baselines**: Static GCN and co-occurrence frequency
- **Constraint**: DHGNN-Lite AUC must exceed co-frequency AUC by ≥0.05
- Trained on 2019–2022, evaluated on 2023 snapshots from the KG

### 8. React Frontend — 3 new research pages

The original frontend only showed task progress. v2 adds three research-facing pages:

- **`/leaderboard`** — sortable benchmark table for all 7 systems, reads from
  `experiments/final/reports_cache.json`
- **`/trace/:paper_id`** — 5-step execution trace viewer: Plan → Parser ensemble →
  Specialist outputs → Critic verdicts (with evidence drill-down) → Reflexion update
- **`/kg`** — interactive 2-hop institution collaboration subgraph using React Flow;
  select any (venue, year) pair; click nodes to inspect properties

---

## Why These Design Choices

1. **Reflexion over fine-tuning**: fine-tuning would require labelled data we don't have
   at run-time. Reflexion lets the system accumulate venue-specific knowledge during
   the benchmark run itself, which is more practical.

2. **Tool-grounded Critic over a second LLM extraction**: using ROR/OpenAlex as ground-truth
   tools gives the Critic external evidence, reducing hallucination compared to a
   purely LLM-based re-checker.

3. **KuzuDB over a hosted graph DB**: the system needs to run embedded for reproducibility.
   KuzuDB provides Cypher query support with no separate server process.

4. **PaperAffilBench as a first-class contribution**: there is no existing public benchmark
   for cross-venue affiliation extraction with hard-case coverage. Creating one lets the
   community evaluate future systems against a reproducible baseline.

---

## Reproducibility

Every result in `experiments/final/REPORT.md` can be reproduced from the cached
predictions in under 30 minutes:

```bash
make repro
```

The full re-run (API calls to 3 LLM providers) is:

```bash
MAX_USD=80 make repro-full
```

---

I am happy to walk through any part of the system in more detail at your convenience.

Best regards
