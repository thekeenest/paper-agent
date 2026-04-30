# src/v2/eval/

**Purpose:** PaperAffilBench evaluation harness, baseline runners, and metric computation.

## PaperAffilBench

| Property | Value |
|---|---|
| Size | ≈800 papers |
| Venues | NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR |
| Years | 3 (e.g., 2022–2024) |
| Annotation | Manual + inter-annotator agreement (κ, α) |
| Location | `benchmark/PaperAffilBench/` |

## Planned modules

| Module | Purpose |
|---|---|
| `harness.py` | Runs any extractor pipeline on the benchmark split |
| `baselines.py` | GROBID, OpenAlex, S2AFF, v1-frozen wrappers |
| `metrics.py` | Author F1, ROR-linking accuracy, ECE, cost, latency |
| `annotator.py` | Annotation ingestion + IAA computation |
| `ablations.py` | Ablation table generator (Tables 1–4 of the coursework) |

See [DEV_PLAN.md §4](../../../coursework_v2/DEV_PLAN.md) for the full evaluation design.
