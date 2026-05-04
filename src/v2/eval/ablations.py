"""
Ablation study runner for Paper-Agent v2.

Runs a grid of system configurations and produces:
  * Table 4.2 — Component ablation (P/R/F1 and ROR/country/type accuracy)
  * Table 4.3 — Confidence calibration ablation (ECE and all_correct)

Both tables are emitted as booktabs LaTeX and as console ASCII.

Ablation grid
-------------
The grid removes one component at a time, comparing against the full pipeline:

  +-------------------------------+--------+--------+---------+-----------+
  | System                        | Planner| Critic | Reflexion| S2AND     |
  +-------------------------------+--------+--------+---------+-----------+
  | full_v2                       |   ✓    |   ✓    |    ✓    |    ✓      |
  | plan_act_critic               |   ✓    |   ✓    |    ✗    |    ✓      |
  | plan_act                      |   ✓    |   ✗    |    ✗    |    ✓      |
  | openalex_pipeline (no planner)|   ✗    |   ✗    |    ✗    |    ✗      |
  | grobid_ror (external baseline)|   ✗    |   ✗    |    ✗    |    ✗      |
  +-------------------------------+--------+--------+---------+-----------+

Usage
-----
::

    python -m src.v2.eval.ablations --split test
    python -m src.v2.eval.ablations --split mini --output output/ablations
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.v2.eval.metrics import MetricsReport
from src.v2.eval.runner import run_evaluation

_OUTPUT_DIR = Path(os.getenv("ABLATION_OUTPUT", "output/ablations"))

# Systems to include in Table 4.2 and 4.3
_TABLE_4_2_SYSTEMS = [
    "grobid_ror",
    "openalex_pipeline",
    "s2aff",
    "plan_act",
    "plan_act_critic",
    "full_v2",
]
_TABLE_4_3_SYSTEMS = [
    "plan_act",
    "plan_act_critic",
    "full_v2",
]


# ─────────────────────────── LaTeX builders ───────────────────────────────────


def _table_4_2(reports: dict[str, MetricsReport]) -> str:
    """Return the LaTeX source for Table 4.2 (component ablation)."""
    rows = []
    for system in _TABLE_4_2_SYSTEMS:
        if system not in reports:
            continue
        r = reports[system]
        label = _system_label(system)
        rows.append(r.to_booktabs_row(label))

    body = "\n".join(rows)
    return rf"""
% Table 4.2 — Component ablation (author-affiliation extraction)
\begin{{table}}[ht]
\centering
\caption{{Component ablation on PaperAffilBench (NeurIPS, test split).
         All metrics are macro-averaged over papers.
         Best result per column in \textbf{{bold}}.}}
\label{{tab:ablation_component}}
\begin{{tabular}}{{lcccccccc}}
\toprule
System & P & R & F1 & ROR-acc & Country-acc & Type-acc & All-correct & ECE \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _table_4_3(reports: dict[str, MetricsReport]) -> str:
    """Return the LaTeX source for Table 4.3 (calibration ablation)."""
    rows = []
    for system in _TABLE_4_3_SYSTEMS:
        if system not in reports:
            continue
        r = reports[system]
        label = _system_label(system)
        # Table 4.3 focuses on calibration + all_correct
        row = (
            f"{label} & {r.f1:.3f} & {r.all_correct:.3f} & {r.ece:.3f} "
            f"& {r.surface_form_f1:.3f} \\\\"
        )
        rows.append(row)

    body = "\n".join(rows)
    return rf"""
% Table 4.3 — Calibration and completeness ablation
\begin{{table}}[ht]
\centering
\caption{{Calibration ablation on PaperAffilBench (NeurIPS, test split).
         ECE = Expected Calibration Error (lower is better).}}
\label{{tab:ablation_calibration}}
\begin{{tabular}}{{lcccc}}
\toprule
System & F1 & All-correct & ECE $\downarrow$ & Surface-F1 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _system_label(system: str) -> str:
    labels = {
        "grobid_ror": r"GROBID + ROR",
        "openalex_pipeline": r"OpenAlex",
        "s2aff": r"S2AFF",
        "v1_frozen": r"v1 (frozen)",
        "plan_act": r"Plan+Act (no Critic)",
        "plan_act_critic": r"Plan+Act+Critic",
        "full_v2": r"\textbf{Full v2} (ours)",
    }
    return labels.get(system, system)


# ─────────────────────────── ASCII tables ────────────────────────────────────


def _ascii_table_4_2(reports: dict[str, MetricsReport]) -> str:
    col_w = 20
    header = f"{'System':<{col_w}} {'P':>6} {'R':>6} {'F1':>6} {'ROR':>6} {'Cntry':>6} {'Type':>6} {'AllOK':>6} {'ECE':>6}"
    sep = "-" * len(header)
    lines = ["\nTable 4.2 — Component ablation", sep, header, sep]
    for system in _TABLE_4_2_SYSTEMS:
        if system not in reports:
            continue
        r = reports[system]
        lines.append(
            f"{system:<{col_w}} {r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f}"
            f" {r.ror_linking_accuracy:>6.3f} {r.country_accuracy:>6.3f}"
            f" {r.type_accuracy:>6.3f} {r.all_correct:>6.3f} {r.ece:>6.3f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _ascii_table_4_3(reports: dict[str, MetricsReport]) -> str:
    col_w = 20
    header = f"{'System':<{col_w}} {'F1':>6} {'AllOK':>6} {'ECE':>6} {'SurfF1':>7}"
    sep = "-" * len(header)
    lines = ["\nTable 4.3 — Calibration ablation", sep, header, sep]
    for system in _TABLE_4_3_SYSTEMS:
        if system not in reports:
            continue
        r = reports[system]
        lines.append(
            f"{system:<{col_w}} {r.f1:>6.3f} {r.all_correct:>6.3f}"
            f" {r.ece:>6.3f} {r.surface_form_f1:>7.3f}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────── runner ──────────────────────────────────────────


def run_ablations(
    split: str = "test",
    systems: list[str] | None = None,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> dict[str, MetricsReport]:
    """Run ablation grid and return dict of system → MetricsReport.

    Parameters
    ----------
    split:
        Benchmark split ("test", "dev", "mini").
    systems:
        List of systems to run.  Defaults to all Table 4.2 + 4.3 systems.
    output_dir:
        Where to write reports and LaTeX.
    verbose:
        Print per-paper progress.
    """
    if systems is None:
        systems = list(dict.fromkeys(_TABLE_4_2_SYSTEMS + _TABLE_4_3_SYSTEMS))

    out_dir = output_dir or _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, MetricsReport] = {}
    for system in systems:
        print(f"\n[Ablation] Running {system} on split={split} ...")
        try:
            report = run_evaluation(
                system=system,
                split=split,
                output_dir=out_dir / "runs",
                verbose=verbose,
            )
            reports[system] = report
            print(f"  F1={report.f1:.3f}  ECE={report.ece:.3f}  run_id={report.run_id}")
        except Exception as exc:
            print(f"  Warning: {system} failed: {exc}")

    # Persist reports
    summary: dict[str, Any] = {s: r.to_dict() for s, r in reports.items()}
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Emit LaTeX
    latex_4_2 = _table_4_2(reports)
    latex_4_3 = _table_4_3(reports)
    (out_dir / "table_4_2.tex").write_text(latex_4_2, encoding="utf-8")
    (out_dir / "table_4_3.tex").write_text(latex_4_3, encoding="utf-8")

    return reports


# ─────────────────────────── CLI ─────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.v2.eval.ablations",
        description="Run ablation grid and produce Tables 4.2 and 4.3.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test", "mini"],
        help="Benchmark split (default: test)",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Systems to run (default: all ablation systems)",
    )
    parser.add_argument("--output", default=str(_OUTPUT_DIR))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    reports = run_ablations(
        split=args.split,
        systems=args.systems,
        output_dir=Path(args.output),
        verbose=args.verbose,
    )

    # Print ASCII tables
    print(_ascii_table_4_2(reports))
    print(_ascii_table_4_3(reports))

    # Print LaTeX
    print("\n\n% ── LaTeX output ──────────────────────────────────────────")
    print(_table_4_2(reports))
    print(_table_4_3(reports))

    out = Path(args.output)
    print(f"\nAblation outputs written to {out}/")
    print(f"  LaTeX: {out}/table_4_2.tex, {out}/table_4_3.tex")
    print(f"  JSON : {out}/ablation_summary.json")


if __name__ == "__main__":
    main()
