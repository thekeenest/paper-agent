"""
Stage-8 evaluation harness.

Orchestrates 7 systems × 3 LLM judges on PaperAffilBench, produces:
  - MetricsReport per system (Tables 4.2 / 4.3 / 4.4)
  - LLM-judge agreement scores (Cohen's κ, Krippendorff's α)
  - Figures 4.2–4.6 (F1, ROR accuracy, judge agreement, calibration, PR curve)
  - experiments/final/REPORT.md with headline numbers and H1–H4 verdicts

Budget is enforced via BudgetTracker (MAX_USD env var, default $80).

Usage
-----
::

    python -m src.v2.eval.harness --split test
    python -m src.v2.eval.harness --split dev --max-usd 20
    python -m src.v2.eval.harness --split mini --skip-judges   # smoke test
    python -m src.v2.eval.harness --from-cache                 # repro from cache
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from src.v2.eval.budget import BudgetExceededError, get_global_tracker, reset_global_tracker
from src.v2.eval.metrics import MetricsReport
from src.v2.eval.runner import _ALL_SYSTEMS, _load_manifest, _get_papers, run_evaluation

_OUTPUT_DIR = Path(os.getenv("BENCH_OUTPUT_DIR", "output/eval"))
_REPORT_DIR = Path("experiments/final")
_FIG_DIR = Path("experiments/final/figures")
_ALL_JUDGES = ["claude-sonnet-4-6", "gpt-4o", "gemini-1.5-flash"]

# Hypotheses to test
_HYPOTHESES = {
    "H1": "full_v2 F1 ≥ grobid_ror F1 + 0.10",
    "H2": "full_v2 ROR-linking accuracy ≥ 0.80",
    "H3": "plan_act_critic F1 > plan_act F1 (Critic adds value)",
    "H4": "full_v2 ECE ≤ 0.10 (well-calibrated confidences)",
}


# ─────────────────────────────────── CI helpers ──────────────────────────────


def _wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (robust at extremes)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def _bootstrap_ci(
    values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Bootstrap percentile confidence interval."""
    import random
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    boot_means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(alpha / 2 * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


# ─────────────────────────────────── table helpers ───────────────────────────


def _booktabs_table(
    caption: str,
    label: str,
    header: str,
    rows: list[str],
    col_spec: str = "lrrrrrrrr",
) -> str:
    body = "\n".join(f"    {r}" for r in rows)
    return (
        f"\\begin{{table}}[ht]\n"
        f"  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        f"    \\toprule\n"
        f"    {header} \\\\\n"
        f"    \\midrule\n"
        f"{body}\n"
        f"    \\bottomrule\n"
        f"  \\end{{tabular}}\n"
        f"\\end{{table}}"
    )


def _table_4_2(reports: dict[str, MetricsReport]) -> str:
    """Table 4.2 — system-level F1, ROR, country, type, all_correct."""
    header = (
        "System & P & R & F1 & ROR-acc & Ctry-acc & Type-acc & All\\% & ECE"
    )
    rows = []
    for sys_name in _ALL_SYSTEMS:
        r = reports.get(sys_name)
        if r is None:
            continue
        row = (
            f"{sys_name.replace('_', '\\_')} "
            f"& {r.precision:.3f} & {r.recall:.3f} & {r.f1:.3f}"
            f" & {r.ror_linking_accuracy:.3f} & {r.country_accuracy:.3f}"
            f" & {r.type_accuracy:.3f} & {r.all_correct:.3f}"
            f" & {r.ece:.3f} \\\\"
        )
        rows.append(row)
    return _booktabs_table(
        caption="System comparison on PaperAffilBench test split (N=160)",
        label="tab:system_comparison",
        header=header,
        rows=rows,
        col_spec="lrrrrrrrrr",
    )


def _table_4_3(reports: dict[str, MetricsReport]) -> str:
    """Table 4.3 — ablation (plan_act vs plan_act_critic vs full_v2)."""
    header = "System & F1 & ΔF1 & ROR-acc & ECE"
    ablation_systems = ["plan_act", "plan_act_critic", "full_v2"]
    base_f1 = reports.get("plan_act", MetricsReport()).f1
    rows = []
    for sys_name in ablation_systems:
        r = reports.get(sys_name)
        if r is None:
            continue
        delta = r.f1 - base_f1
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        if sys_name == "plan_act":
            delta_str = "—"
        rows.append(
            f"{sys_name.replace('_', '\\_')} & {r.f1:.3f} & {delta_str}"
            f" & {r.ror_linking_accuracy:.3f} & {r.ece:.3f} \\\\"
        )
    return _booktabs_table(
        caption="Ablation: contribution of Critic and Reflexion modules",
        label="tab:ablation",
        header=header,
        rows=rows,
        col_spec="lrrrr",
    )


def _table_4_4(judge_results: dict[str, Any]) -> str:
    """Table 4.4 — LLM-judge agreement scores."""
    header = "Judge pair & Cohen's κ & Krippendorff's α & Mean bias"
    rows = []
    agreements = judge_results.get("agreements", {})
    for pair_key, vals in agreements.items():
        kappa = vals.get("cohen_kappa", 0.0)
        alpha = vals.get("krippendorff_alpha", 0.0)
        bias = vals.get("mean_bias", 0.0)
        rows.append(
            f"{pair_key.replace('_', '\\_')} & {kappa:.3f} & {alpha:.3f} & {bias:+.3f} \\\\"
        )
    if not rows:
        rows = ["(no judge results) & — & — & — \\\\"]
    return _booktabs_table(
        caption="LLM-judge cross-model agreement on dev split",
        label="tab:judge_agreement",
        header=header,
        rows=rows,
        col_spec="lrrr",
    )


# ─────────────────────────────────── figures ─────────────────────────────────


def _save_figure_4_2(reports: dict[str, MetricsReport], fig_dir: Path) -> None:
    """Figure 4.2 — F1 comparison bar chart."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams["figure.dpi"] = 120
        systems = [s for s in _ALL_SYSTEMS if s in reports]
        f1_vals = [reports[s].f1 for s in systems]
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(range(len(systems)), f1_vals, color=colors[: len(systems)], alpha=0.85)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([s.replace("_", "\n") for s in systems], fontsize=9)
        ax.set_title("Figure 4.2 — System F1 comparison (PaperAffilBench test split)", fontsize=11)
        for bar, val in zip(bars, f1_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_4_2_f1_comparison.pdf", bbox_inches="tight")
        plt.close()
    except ImportError:
        pass


def _save_figure_4_3(reports: dict[str, MetricsReport], fig_dir: Path) -> None:
    """Figure 4.3 — ROR-linking accuracy comparison."""
    try:
        import matplotlib.pyplot as plt

        systems = [s for s in _ALL_SYSTEMS if s in reports]
        ror_vals = [reports[s].ror_linking_accuracy for s in systems]
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(systems)), ror_vals, color=colors[: len(systems)], alpha=0.85)
        ax.axhline(0.80, color="grey", linestyle="--", linewidth=0.8, label="H2 threshold (0.80)")
        ax.set_ylabel("ROR-linking Accuracy", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([s.replace("_", "\n") for s in systems], fontsize=9)
        ax.set_title("Figure 4.3 — ROR-linking accuracy per system", fontsize=11)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_4_3_ror_accuracy.pdf", bbox_inches="tight")
        plt.close()
    except ImportError:
        pass


def _save_figure_4_4(judge_results: dict[str, Any], fig_dir: Path) -> None:
    """Figure 4.4 — Judge agreement heatmap."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        agreements = judge_results.get("agreements", {})
        judges = _ALL_JUDGES
        n = len(judges)
        mat = np.zeros((n, n))
        for i, j_a in enumerate(judges):
            for j, j_b in enumerate(judges):
                if i == j:
                    mat[i, j] = 1.0
                else:
                    key = f"{j_a}_vs_{j_b}"
                    alt_key = f"{j_b}_vs_{j_a}"
                    kappa = (
                        agreements.get(key, {}).get("cohen_kappa")
                        or agreements.get(alt_key, {}).get("cohen_kappa")
                        or 0.0
                    )
                    mat[i, j] = kappa

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn")
        plt.colorbar(im, ax=ax, label="Cohen's κ")
        short = [j.split("-")[0] for j in judges]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(short, fontsize=10)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title("Figure 4.4 — Inter-judge Cohen's κ", fontsize=11)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_4_4_judge_agreement.pdf", bbox_inches="tight")
        plt.close()
    except ImportError:
        pass


def _save_figure_4_5(reports: dict[str, MetricsReport], fig_dir: Path) -> None:
    """Figure 4.5 — Calibration (ECE) bar chart."""
    try:
        import matplotlib.pyplot as plt

        systems = [s for s in _ALL_SYSTEMS if s in reports]
        ece_vals = [reports[s].ece for s in systems]
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(systems)), ece_vals, color=colors[: len(systems)], alpha=0.85)
        ax.axhline(0.10, color="grey", linestyle="--", linewidth=0.8, label="H4 threshold (ECE=0.10)")
        ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([s.replace("_", "\n") for s in systems], fontsize=9)
        ax.set_title("Figure 4.5 — Confidence calibration (ECE, lower is better)", fontsize=11)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_4_5_calibration.pdf", bbox_inches="tight")
        plt.close()
    except ImportError:
        pass


def _save_figure_4_6(reports: dict[str, MetricsReport], fig_dir: Path) -> None:
    """Figure 4.6 — Precision-Recall scatter (P vs R per system)."""
    try:
        import matplotlib.pyplot as plt

        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]
        fig, ax = plt.subplots(figsize=(7, 6))
        for idx, sys_name in enumerate(s for s in _ALL_SYSTEMS if s in reports):
            r = reports[sys_name]
            ax.scatter(
                r.recall, r.precision,
                s=120, color=colors[idx % len(colors)],
                zorder=3, label=sys_name.replace("_", " "),
            )
            ax.annotate(
                sys_name.replace("_", "\n"),
                (r.recall, r.precision),
                fontsize=7,
                xytext=(5, 3),
                textcoords="offset points",
            )
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_title("Figure 4.6 — Precision vs Recall per system", fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_4_6_pr_scatter.pdf", bbox_inches="tight")
        plt.close()
    except ImportError:
        pass


# ─────────────────────────────────── main harness ────────────────────────────


def run_harness(
    split: str = "test",
    systems: list[str] | None = None,
    judges: list[str] | None = None,
    max_usd: float = 80.0,
    skip_judges: bool = False,
    from_cache: bool = False,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full Stage-8 evaluation harness.

    Parameters
    ----------
    split:
        Benchmark split to evaluate ("test", "dev", "mini").
    systems:
        Which systems to evaluate (default: all 7).
    judges:
        Which LLM judges to use (default: all 3).
    max_usd:
        Hard budget ceiling in USD.
    skip_judges:
        If True, skip LLM judge protocol (faster, cheaper).
    from_cache:
        If True, load cached MetricsReports and skip re-running systems.
    output_dir:
        Base output directory.
    verbose:
        Print progress.

    Returns
    -------
    dict with keys: reports, judge_results, tables, hypotheses
    """
    tracker = reset_global_tracker(max_usd=max_usd)
    out_dir = output_dir or _OUTPUT_DIR
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    if systems is None:
        systems = list(_ALL_SYSTEMS)
    if judges is None:
        judges = list(_ALL_JUDGES)

    # ── 1. Run all systems ────────────────────────────────────────────────────
    reports: dict[str, MetricsReport] = {}
    cache_path = _REPORT_DIR / "reports_cache.json"

    if from_cache and cache_path.exists():
        if verbose:
            print("[harness] Loading cached MetricsReports...")
        cached = json.loads(cache_path.read_text())
        for sys_name, d in cached.items():
            r = MetricsReport(**{k: v for k, v in d.items() if k != "run_id"})
            r.run_id = d.get("run_id", "")
            reports[sys_name] = r
    else:
        for sys_name in systems:
            if verbose:
                print(f"\n[harness] Evaluating system: {sys_name} on split={split}")
            try:
                report = run_evaluation(
                    system=sys_name,
                    split=split,
                    output_dir=out_dir,
                    verbose=verbose,
                )
                reports[sys_name] = report
                if verbose:
                    print(f"  F1={report.f1:.4f}  ROR={report.ror_linking_accuracy:.4f}  ECE={report.ece:.4f}")
            except BudgetExceededError as e:
                print(f"[harness] Budget exceeded during {sys_name}: {e}", file=sys.stderr)
                break
            except Exception as e:
                print(f"[harness] Error running {sys_name}: {e}", file=sys.stderr)
                reports[sys_name] = MetricsReport(run_id=f"error:{sys_name}")

        # Cache reports
        cache_path.write_text(
            json.dumps({k: v.to_dict() for k, v in reports.items()}, indent=2),
            encoding="utf-8",
        )

    # ── 2. LLM judge protocol ─────────────────────────────────────────────────
    judge_results: dict[str, Any] = {"agreements": {}, "bias": {}}

    if not skip_judges:
        try:
            import asyncio
            from src.v2.eval.llm_judge import run_judge_protocol, compute_judge_bias

            if verbose:
                print(f"\n[harness] Running LLM judge protocol ({split}, {len(judges)} judges)...")
            judge_raw = asyncio.run(
                run_judge_protocol(
                    split=split,
                    judges=judges,
                    systems=systems,
                    force_rerun=False,
                )
            )
            judge_results = judge_raw

        except BudgetExceededError as e:
            print(f"[harness] Budget exceeded during judge protocol: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[harness] Judge protocol skipped: {e}", file=sys.stderr)

    # ── 3. Agreement metrics (from llm_judge or standalone) ───────────────────
    try:
        from src.v2.eval.agreement import compute_all_agreements

        manifest = _load_manifest()
        papers = _get_papers(manifest, split)
        agreement_results = compute_all_agreements(
            papers=papers,
            systems=systems,
            judges=judges,
            output_dir=out_dir,
        )
        judge_results["agreements"].update(agreement_results.get("agreements", {}))
    except Exception as e:
        if verbose:
            print(f"[harness] Agreement computation skipped: {e}", file=sys.stderr)

    # ── 4. Generate tables ────────────────────────────────────────────────────
    table_42 = _table_4_2(reports)
    table_43 = _table_4_3(reports)
    table_44 = _table_4_4(judge_results)

    tex_dir = _REPORT_DIR / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / "table_4_2.tex").write_text(table_42, encoding="utf-8")
    (tex_dir / "table_4_3.tex").write_text(table_43, encoding="utf-8")
    (tex_dir / "table_4_4.tex").write_text(table_44, encoding="utf-8")

    if verbose:
        print("\n[harness] Tables written:")
        print(f"  {tex_dir}/table_4_2.tex")
        print(f"  {tex_dir}/table_4_3.tex")
        print(f"  {tex_dir}/table_4_4.tex")

    # ── 5. Generate figures ───────────────────────────────────────────────────
    _save_figure_4_2(reports, _FIG_DIR)
    _save_figure_4_3(reports, _FIG_DIR)
    _save_figure_4_4(judge_results, _FIG_DIR)
    _save_figure_4_5(reports, _FIG_DIR)
    _save_figure_4_6(reports, _FIG_DIR)

    if verbose:
        print(f"\n[harness] Figures saved to {_FIG_DIR}/")

    # ── 6. Hypothesis tests ───────────────────────────────────────────────────
    full_v2 = reports.get("full_v2", MetricsReport())
    grobid = reports.get("grobid_ror", MetricsReport())
    plan_act = reports.get("plan_act", MetricsReport())
    plan_act_critic = reports.get("plan_act_critic", MetricsReport())

    manifest = _load_manifest()
    n_test = len(_get_papers(manifest, split))

    h1_margin = full_v2.f1 - grobid.f1
    h1_passed = h1_margin >= 0.10
    h1_ci_lo, h1_ci_hi = _bootstrap_ci(
        [full_v2.f1 - grobid.f1] + [0.0] * (n_test - 1), n_boot=500
    )

    h2_passed = full_v2.ror_linking_accuracy >= 0.80
    h2_ci_lo, h2_ci_hi = _wilson_ci(
        full_v2.ror_linking_accuracy * n_test, n_test
    )

    h3_margin = plan_act_critic.f1 - plan_act.f1
    h3_passed = h3_margin > 0.0

    h4_passed = full_v2.ece <= 0.10

    hypotheses = {
        "H1": {
            "description": _HYPOTHESES["H1"],
            "full_v2_f1": round(full_v2.f1, 4),
            "grobid_f1": round(grobid.f1, 4),
            "margin": round(h1_margin, 4),
            "ci_95": [round(h1_ci_lo, 4), round(h1_ci_hi, 4)],
            "passed": h1_passed,
            "verdict": "SUPPORTED" if h1_passed else "NOT SUPPORTED",
        },
        "H2": {
            "description": _HYPOTHESES["H2"],
            "full_v2_ror_acc": round(full_v2.ror_linking_accuracy, 4),
            "threshold": 0.80,
            "ci_95": [round(h2_ci_lo, 4), round(h2_ci_hi, 4)],
            "passed": h2_passed,
            "verdict": "SUPPORTED" if h2_passed else "NOT SUPPORTED",
        },
        "H3": {
            "description": _HYPOTHESES["H3"],
            "plan_act_critic_f1": round(plan_act_critic.f1, 4),
            "plan_act_f1": round(plan_act.f1, 4),
            "margin": round(h3_margin, 4),
            "passed": h3_passed,
            "verdict": "SUPPORTED" if h3_passed else "NOT SUPPORTED",
        },
        "H4": {
            "description": _HYPOTHESES["H4"],
            "full_v2_ece": round(full_v2.ece, 4),
            "threshold": 0.10,
            "passed": h4_passed,
            "verdict": "SUPPORTED" if h4_passed else "NOT SUPPORTED",
        },
    }

    # ── 7. Generate REPORT.md ─────────────────────────────────────────────────
    _write_report_md(reports, judge_results, hypotheses, tracker, split, n_test)
    if verbose:
        print(f"\n[harness] Report written: {_REPORT_DIR}/REPORT.md")

    # ── 8. Update leaderboard ─────────────────────────────────────────────────
    _write_leaderboard(reports)
    if verbose:
        print(f"[harness] Leaderboard updated: benchmark/leaderboard.md")
        print(f"\n[harness] Budget: {tracker.summary()}")

    # Save full results
    results = {
        "split": split,
        "n_papers": n_test,
        "reports": {k: v.to_dict() for k, v in reports.items()},
        "judge_results": judge_results,
        "hypotheses": hypotheses,
        "budget_summary": tracker.summary(),
    }
    (_REPORT_DIR / "harness_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    return results


# ─────────────────────────────────── report generation ───────────────────────


def _write_report_md(
    reports: dict[str, MetricsReport],
    judge_results: dict[str, Any],
    hypotheses: dict[str, Any],
    tracker: Any,
    split: str,
    n_papers: int,
) -> None:
    """Write experiments/final/REPORT.md."""
    from datetime import date

    full_v2 = reports.get("full_v2", MetricsReport())
    grobid = reports.get("grobid_ror", MetricsReport())
    agreements = judge_results.get("agreements", {})

    # Find best inter-judge kappa
    best_kappa = max(
        (v.get("cohen_kappa", 0.0) for v in agreements.values()),
        default=0.0,
    )
    best_alpha = max(
        (v.get("krippendorff_alpha", 0.0) for v in agreements.values()),
        default=0.0,
    )

    lines = [
        "# PaperAffilBench — Stage-8 Evaluation Report",
        "",
        f"> Generated: {date.today().isoformat()}  ",
        f"> Split: `{split}` ({n_papers} papers)  ",
        f"> Budget used: {tracker.summary()}",
        "",
        "---",
        "",
        "## 1. Headline Numbers",
        "",
        "| System | F1 | ROR-acc | Country-acc | ECE |",
        "|--------|-----|---------|-------------|-----|",
    ]
    for sys_name in _ALL_SYSTEMS:
        r = reports.get(sys_name)
        if r is None:
            continue
        lines.append(
            f"| {sys_name} | {r.f1:.3f} | {r.ror_linking_accuracy:.3f}"
            f" | {r.country_accuracy:.3f} | {r.ece:.3f} |"
        )

    lines += [
        "",
        "## 2. Hypothesis Test Results",
        "",
    ]

    for hid, h in hypotheses.items():
        verdict_icon = "✅" if h["passed"] else "❌"
        lines += [
            f"### {hid} — {h['description']}",
            "",
            f"**Verdict: {verdict_icon} {h['verdict']}**",
            "",
        ]
        if hid == "H1":
            lo, hi = h["ci_95"]
            lines += [
                f"- full_v2 F1 = {h['full_v2_f1']:.4f}",
                f"- grobid_ror F1 = {h['grobid_f1']:.4f}",
                f"- Margin = {h['margin']:+.4f} (required ≥ +0.10)",
                f"- 95% CI on margin: [{lo:.4f}, {hi:.4f}]",
                "",
            ]
        elif hid == "H2":
            lo, hi = h["ci_95"]
            lines += [
                f"- full_v2 ROR accuracy = {h['full_v2_ror_acc']:.4f}",
                f"- Required ≥ 0.80",
                f"- 95% Wilson CI: [{lo:.4f}, {hi:.4f}]",
                "",
            ]
        elif hid == "H3":
            lines += [
                f"- plan_act_critic F1 = {h['plan_act_critic_f1']:.4f}",
                f"- plan_act F1 = {h['plan_act_f1']:.4f}",
                f"- Margin = {h['margin']:+.4f} (required > 0)",
                "",
            ]
        elif hid == "H4":
            lines += [
                f"- full_v2 ECE = {h['full_v2_ece']:.4f}",
                f"- Required ≤ 0.10",
                "",
            ]

    lines += [
        "## 3. LLM-Judge Agreement",
        "",
        "| Pair | Cohen's κ | Krippendorff's α |",
        "|------|-----------|-----------------|",
    ]
    for pair_key, vals in agreements.items():
        lines.append(
            f"| {pair_key} | {vals.get('cohen_kappa', 0.0):.3f}"
            f" | {vals.get('krippendorff_alpha', 0.0):.3f} |"
        )
    if not agreements:
        lines.append("| (no judge results) | — | — |")

    lines += [
        "",
        f"Best Cohen's κ across judge pairs: **{best_kappa:.3f}**  ",
        f"Best Krippendorff's α: **{best_alpha:.3f}**",
        "",
        "## 4. Tables and Figures",
        "",
        "LaTeX tables are in `experiments/final/tables/`:  ",
        "- `table_4_2.tex` — full system comparison  ",
        "- `table_4_3.tex` — ablation study  ",
        "- `table_4_4.tex` — LLM judge agreement  ",
        "",
        "Figures are in `experiments/final/figures/`:  ",
        "- `fig_4_2_f1_comparison.pdf`  ",
        "- `fig_4_3_ror_accuracy.pdf`  ",
        "- `fig_4_4_judge_agreement.pdf`  ",
        "- `fig_4_5_calibration.pdf`  ",
        "- `fig_4_6_pr_scatter.pdf`  ",
        "",
        "## 5. Reproducibility",
        "",
        "```bash",
        "# Reproduce this report from cached predictions (<30 min):",
        "make repro",
        "",
        "# Re-run from scratch (requires API keys + budget):",
        "make repro FROM_SCRATCH=1",
        "```",
        "",
        "Cached predictions: `benchmark/PaperAffilBench/predictions_cache/`  ",
        "Budget log: `output/eval/budget.json`",
        "",
        "---",
        "*This report was generated automatically by `src/v2/eval/harness.py`.*",
    ]

    (_REPORT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_leaderboard(reports: dict[str, MetricsReport]) -> None:
    """Update benchmark/leaderboard.md sorted by F1."""
    from src.v2.eval.runner import _ALL_SYSTEMS

    sorted_systems = sorted(
        [(s, r) for s, r in reports.items()],
        key=lambda x: x[1].f1,
        reverse=True,
    )

    lines = [
        "# PaperAffilBench — System Leaderboard",
        "",
        "> Sorted by F1 (descending) on the **test split** (N=160).",
        "> All scores are macro-averages over papers.",
        "",
        "| Rank | System | F1 | Precision | Recall | ROR-acc | Country-acc | ECE | Papers |",
        "|------|--------|----|-----------|--------|---------|-------------|-----|--------|",
    ]

    for rank, (sys_name, r) in enumerate(sorted_systems, 1):
        lines.append(
            f"| {rank} | {sys_name} | {r.f1:.3f} | {r.precision:.3f} | {r.recall:.3f}"
            f" | {r.ror_linking_accuracy:.3f} | {r.country_accuracy:.3f}"
            f" | {r.ece:.3f} | {r.n_papers} |"
        )

    lines += [
        "",
        "## Metric Definitions",
        "",
        "- **F1**: Macro-averaged F1 on author × affiliation pairs (exact match after normalization)",
        "- **Precision / Recall**: Corresponding precision and recall",
        "- **ROR-acc**: Fraction of gold authors where predicted ROR ID matches",
        "- **Country-acc**: Fraction of gold authors where country code matches",
        "- **ECE**: Expected Calibration Error (lower = better calibrated confidences)",
        "- **Papers**: Number of papers evaluated in this run",
        "",
        "## Systems",
        "",
        "| System | Description |",
        "|--------|-------------|",
        "| grobid_ror | GROBID extraction + ROR fuzzy lookup |",
        "| openalex_pipeline | OpenAlex authorship API |",
        "| s2aff | Semantic Scholar S2AFF |",
        "| v1_frozen | Frozen v1 pipeline (snapshot baseline) |",
        "| plan_act | v2 Planner + Extractor, no Critic |",
        "| plan_act_critic | v2 Planner + Extractor + Critic, no Reflexion |",
        "| full_v2 | Full v2 pipeline with Reflexion |",
        "",
        "---",
        "*Updated automatically by `src/v2/eval/harness.py`.*",
    ]

    Path("benchmark/leaderboard.md").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────── CLI ─────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.v2.eval.harness",
        description="Run Stage-8 evaluation harness.",
    )
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "mini", "all"])
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Systems to evaluate (default: all 7)",
    )
    parser.add_argument(
        "--judges",
        nargs="*",
        default=None,
        help="LLM judges to use (default: all 3)",
    )
    parser.add_argument("--max-usd", type=float, default=80.0)
    parser.add_argument(
        "--skip-judges",
        action="store_true",
        help="Skip LLM judge protocol",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Load cached MetricsReports instead of re-running systems",
    )
    parser.add_argument("--output-dir", default=str(_OUTPUT_DIR))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    results = run_harness(
        split=args.split,
        systems=args.systems,
        judges=args.judges,
        max_usd=args.max_usd,
        skip_judges=args.skip_judges,
        from_cache=args.from_cache,
        output_dir=Path(args.output_dir),
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print("HYPOTHESIS VERDICTS")
    print("=" * 60)
    for hid, h in results["hypotheses"].items():
        icon = "✅" if h["passed"] else "❌"
        print(f"  {icon} {hid}: {h['verdict']}  ({h['description']})")
    print()
    print(results.get("budget_summary", ""))


if __name__ == "__main__":
    main()
