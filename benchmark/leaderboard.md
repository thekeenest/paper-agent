# PaperAffilBench — System Leaderboard

> Sorted by F1 (descending) on the **test split** (N=160).
> All scores are macro-averages over papers.

| Rank | System | F1 | Precision | Recall | ROR-acc | Country-acc | ECE | Papers |
|------|--------|----|-----------|--------|---------|-------------|-----|--------|
| 1 | full_v2 | 0.805 | 0.820 | 0.790 | 0.840 | 0.910 | 0.070 | 160 |
| 2 | plan_act_critic | 0.780 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160 |
| 3 | plan_act | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160 |
| 4 | grobid_ror | 0.695 | 0.680 | 0.710 | 0.720 | 0.850 | 0.150 | 160 |

## Metric Definitions

- **F1**: Macro-averaged F1 on author × affiliation pairs (exact match after normalization)
- **Precision / Recall**: Corresponding precision and recall
- **ROR-acc**: Fraction of gold authors where predicted ROR ID matches
- **Country-acc**: Fraction of gold authors where country code matches
- **ECE**: Expected Calibration Error (lower = better calibrated confidences)
- **Papers**: Number of papers evaluated in this run

## Systems

| System | Description |
|--------|-------------|
| grobid_ror | GROBID extraction + ROR fuzzy lookup |
| openalex_pipeline | OpenAlex authorship API |
| s2aff | Semantic Scholar S2AFF |
| v1_frozen | Frozen v1 pipeline (snapshot baseline) |
| plan_act | v2 Planner + Extractor, no Critic |
| plan_act_critic | v2 Planner + Extractor + Critic, no Reflexion |
| full_v2 | Full v2 pipeline with Reflexion |

---
*Updated automatically by `src/v2/eval/harness.py`.*