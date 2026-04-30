"""
Evaluation layer — PaperAffilBench harness, baselines, and metrics.

PaperAffilBench: ~800 manually verified papers from 6 venues × 3 years.
Inter-annotator agreement: Cohen's κ and Krippendorff's α.

Baselines wired:
  - GROBID + ROR API
  - OpenAlex pipeline
  - S2AFF
  - v1-frozen (this repo, src/v1/)
  - Plan+Act (no critic)
  - Plan+Act+Critic
  - Full v2 (Plan+Act+Critic+Reflexion)

Metrics (per DEV_PLAN.md §4):
  author_f1, affil_surface_f1, ror_linking_acc,
  country_acc, org_type_acc, paper_all_correct,
  calibration_ece, cost_per_paper, latency_p50

Status: STUB — no implementation yet.
"""
