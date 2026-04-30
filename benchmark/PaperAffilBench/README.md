# PaperAffilBench

A manually verified benchmark for author–affiliation extraction at conference scale.

## Structure

```
PaperAffilBench/
├── papers/   # Raw PDFs (gitignored; downloaded by scripts in src/v2/eval/)
├── gold/     # Gold-standard annotation JSON files, one per paper
└── splits/   # train/dev/test split manifests (paper IDs + venue + year)
```

## Target composition

| Venue | Papers | Years |
|---|---|---|
| NeurIPS | ~133 | 2022–2024 |
| ICML | ~133 | 2022–2024 |
| ICLR | ~133 | 2022–2024 |
| ACL | ~133 | 2022–2024 |
| EMNLP | ~133 | 2022–2024 |
| CVPR | ~135 | 2022–2024 |
| **Total** | **≈800** | |

## Gold annotation schema

See `src/v2/eval/annotator.py` (planned) and `docs/GOLD_STANDARD_GUIDE.md`.

Each gold file records: `paper_id`, `venue`, `year`, `authors[]` where each author has
`name`, `raw_affiliation`, `ror_id`, `country_code`, `org_type`, `confidence`.

## Inter-annotator agreement

Target: Cohen's κ ≥ 0.80 on `ror_id` assignment, Krippendorff's α ≥ 0.75 on `org_type`.

## Status

**EMPTY** — annotation sprint begins in Week 10 per DEV_PLAN.md milestone schedule.
