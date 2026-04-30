# experiments/

Reproducible experiment artefacts — configs, results CSVs, and ablation tables.

Each subdirectory corresponds to one experiment run:

```
experiments/
├── exp001_baseline_grobid/
│   ├── config.yaml
│   ├── results.parquet
│   └── metrics.json
├── exp002_v1_frozen/
│   └── ...
└── exp003_v2_plan_act/
    └── ...
```

Run experiments via:
```bash
make repro EXP=exp001_baseline_grobid
```

(The `make repro` target is a placeholder — implementation follows DEV_PLAN.md Week 11.)
