"""
Inter-annotator and inter-judge agreement metrics.

Computes Cohen's κ and Krippendorff's α for:
  - Human annotator pairs (A vs B gold annotations)
  - LLM judge pairs (from cached predictions_cache/)

Usage
-----
::

    python -m src.v2.eval.agreement --split dev
    python -m src.v2.eval.agreement --split dev --judges claude gpt4o gemini
"""
from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any

_GOLD_DIR = Path(os.getenv("BENCH_GOLD_DIR", "benchmark/PaperAffilBench/gold"))
_CACHE_DIR = Path(
    os.getenv("JUDGE_CACHE_DIR", "benchmark/PaperAffilBench/predictions_cache")
)
_MANIFEST_PATH = Path(
    os.getenv("BENCH_MANIFEST", "benchmark/PaperAffilBench/manifest.json")
)

_ALL_JUDGES = ["claude-sonnet-4-6", "gpt-4o", "gemini-1.5-flash"]
_JUDGE_SHORT = {
    "claude-sonnet-4-6": "claude",
    "gpt-4o": "gpt4o",
    "gemini-1.5-flash": "gemini",
}


# ─────────────────────────────────── agreement math ──────────────────────────


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's κ for two equal-length ordinal label sequences.

    Labels are integers (e.g. 0, 1, 2 for wrong/partial/correct).
    Returns κ ∈ [-1, 1].  Returns 0.0 for empty or constant sequences.
    """
    n = len(labels_a)
    if n == 0 or len(labels_b) != n:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    k = len(categories)
    if k == 1:
        return 1.0  # perfect agreement by definition (only one category)

    idx = {c: i for i, c in enumerate(categories)}

    # Confusion matrix
    mat = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        mat[idx[a]][idx[b]] += 1

    # Observed agreement
    p_o = sum(mat[i][i] for i in range(k)) / n

    # Expected agreement
    row_sums = [sum(mat[i]) for i in range(k)]
    col_sums = [sum(mat[r][c] for r in range(k)) for c in range(k)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)

    if abs(1.0 - p_e) < 1e-9:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def krippendorff_alpha(ratings: list[list[int | None]]) -> float:
    """Compute Krippendorff's α for ordinal data with multiple raters.

    Parameters
    ----------
    ratings:
        List of rater sequences; ``ratings[r][u]`` is the score rater ``r``
        gave to unit ``u``, or ``None`` for missing data.

    Returns
    -------
    float
        α ∈ [-1, 1].  Returns 0.0 for degenerate inputs.
    """
    n_raters = len(ratings)
    if n_raters < 2:
        return 0.0
    n_units = max(len(r) for r in ratings)
    if n_units == 0:
        return 0.0

    # Normalise to same length
    padded: list[list[int | None]] = []
    for r in ratings:
        padded.append(list(r) + [None] * (n_units - len(r)))

    # Collect all values
    all_vals = [v for row in padded for v in row if v is not None]
    if not all_vals:
        return 0.0
    categories = sorted(set(all_vals))
    if len(categories) == 1:
        return 1.0

    # Value frequencies (for expected disagreement)
    total = len(all_vals)
    freq: dict[int, float] = {c: all_vals.count(c) / total for c in categories}

    # Ordinal distance function: d(c, c') = (c - c')² normalised
    def dist(c: int, cp: int) -> float:
        # Weighted ordinal distance (Krippendorff 2011)
        cats = categories
        ic, icp = cats.index(c), cats.index(cp)
        lo, hi = min(ic, icp), max(ic, icp)
        if lo == hi:
            return 0.0
        # Sum of frequencies in the interval [lo, hi] (inclusive)
        interval_freq = sum(freq[cats[i]] for i in range(lo, hi + 1))
        mid_lo = freq[cats[lo]] / 2
        mid_hi = freq[cats[hi]] / 2
        g = interval_freq - mid_lo - mid_hi
        return g * g

    # Observed disagreement
    d_o = 0.0
    n_pairs = 0
    for u in range(n_units):
        unit_vals = [padded[r][u] for r in range(n_raters) if padded[r][u] is not None]
        m_u = len(unit_vals)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(i + 1, m_u):
                d_o += dist(unit_vals[i], unit_vals[j])
                n_pairs += 1

    if n_pairs == 0:
        return 0.0

    d_o /= n_pairs

    # Expected disagreement
    d_e = 0.0
    for c in categories:
        for cp in categories:
            if c <= cp:
                d_e += freq[c] * freq[cp] * dist(c, cp)
    # Symmetrise
    d_e_full = sum(freq[c] * freq[cp] * dist(c, cp) for c in categories for cp in categories)

    if abs(d_e_full) < 1e-9:
        return 1.0
    return 1.0 - d_o / d_e_full


# ─────────────────────────────────── human annotator agreement ───────────────


def compute_human_agreement(gold_dir: Path = _GOLD_DIR) -> dict[str, Any]:
    """Compute inter-annotator agreement between annotator A and B.

    Looks for pairs of ``<doi>.json`` and ``<doi>.b.json`` in gold_dir.
    Scores are 1 (all correct) or 0 (not), aggregated per paper.
    """
    a_labels: list[int] = []
    b_labels: list[int] = []
    n_pairs = 0

    for a_path in sorted(gold_dir.glob("*.json")):
        if a_path.name.endswith(".b.json"):
            continue
        b_path = a_path.with_suffix("").with_suffix("") if a_path.name.endswith(".a.json") else None
        b_path = gold_dir / (a_path.stem + ".b.json")
        if not b_path.exists():
            continue

        try:
            ann_a = json.loads(a_path.read_text()).get("annotations", [])
            ann_b = json.loads(b_path.read_text()).get("annotations", [])
        except Exception:
            continue

        # Score: fraction of matching author-aff pairs (discretised to 0/1/2)
        gold_set_a = {(a.get("author_name", "").lower(), a.get("normalized_aff", "").lower()) for a in ann_a}
        gold_set_b = {(a.get("author_name", "").lower(), a.get("normalized_aff", "").lower()) for a in ann_b}

        if not gold_set_a and not gold_set_b:
            continue

        overlap = len(gold_set_a & gold_set_b)
        union = len(gold_set_a | gold_set_b)
        jaccard = overlap / union if union > 0 else 1.0

        # Discretise to 0/1/2
        score = 0 if jaccard < 0.5 else (1 if jaccard < 1.0 else 2)
        a_labels.append(2)  # annotator A always assigns full agreement with self
        b_labels.append(score)
        n_pairs += 1

    kappa = cohen_kappa(a_labels, b_labels)
    alpha = krippendorff_alpha([a_labels, b_labels])

    return {
        "n_pairs": n_pairs,
        "cohen_kappa": round(kappa, 4),
        "krippendorff_alpha": round(alpha, 4),
    }


# ─────────────────────────────────── LLM judge agreement ─────────────────────


def _load_judge_labels(
    cache_dir: Path,
    judge: str,
    system: str,
    doi_list: list[str],
) -> list[int | None]:
    """Load cached judge scores for a list of DOIs."""
    labels: list[int | None] = []
    for doi in doi_list:
        safe = doi.replace("/", "_").replace(".", "_")
        path = cache_dir / f"{safe}__{judge}__{system}.json"
        if not path.exists():
            labels.append(None)
            continue
        try:
            data = json.loads(path.read_text())
            # overall_quality is 0/1/2
            labels.append(int(data.get("overall_quality", 1)))
        except Exception:
            labels.append(None)
    return labels


def compute_judge_pair_agreement(
    cache_dir: Path,
    judge_a: str,
    judge_b: str,
    system: str,
    doi_list: list[str],
) -> dict[str, float]:
    """Compute agreement between two judges on a system's predictions."""
    labels_a = _load_judge_labels(cache_dir, judge_a, system, doi_list)
    labels_b = _load_judge_labels(cache_dir, judge_b, system, doi_list)

    # Keep only units where both have a score
    paired_a = [a for a, b in zip(labels_a, labels_b) if a is not None and b is not None]
    paired_b = [b for a, b in zip(labels_a, labels_b) if a is not None and b is not None]

    if len(paired_a) < 2:
        return {"cohen_kappa": 0.0, "krippendorff_alpha": 0.0, "n": 0}

    kappa = cohen_kappa(paired_a, paired_b)
    alpha = krippendorff_alpha([paired_a, paired_b])
    mean_bias = sum(a - b for a, b in zip(paired_a, paired_b)) / len(paired_a)

    return {
        "cohen_kappa": round(kappa, 4),
        "krippendorff_alpha": round(alpha, 4),
        "mean_bias": round(mean_bias, 4),
        "n": len(paired_a),
    }


def compute_all_agreements(
    papers: list[dict[str, Any]],
    systems: list[str],
    judges: list[str] = _ALL_JUDGES,
    cache_dir: Path = _CACHE_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute all pairwise judge agreements across all systems.

    Returns
    -------
    dict with "agreements" key mapping pair_key → agreement scores.
    """
    doi_list = [p["doi"] for p in papers]
    agreements: dict[str, Any] = {}

    for sys_name in systems:
        for j_a, j_b in combinations(judges, 2):
            key = f"{j_a}_vs_{j_b}__{sys_name}"
            result = compute_judge_pair_agreement(cache_dir, j_a, j_b, sys_name, doi_list)
            agreements[key] = result

    # Also compute macro-average across systems per judge pair
    for j_a, j_b in combinations(judges, 2):
        pair_key = f"{j_a}_vs_{j_b}"
        sys_agreements = [
            agreements[f"{j_a}_vs_{j_b}__{s}"]
            for s in systems
            if f"{j_a}_vs_{j_b}__{s}" in agreements
        ]
        if sys_agreements:
            all_kappas = [a["cohen_kappa"] for a in sys_agreements if a["n"] > 0]
            all_alphas = [a["krippendorff_alpha"] for a in sys_agreements if a["n"] > 0]
            all_bias = [a["mean_bias"] for a in sys_agreements if a["n"] > 0]
            agreements[pair_key] = {
                "cohen_kappa": round(sum(all_kappas) / len(all_kappas), 4) if all_kappas else 0.0,
                "krippendorff_alpha": round(sum(all_alphas) / len(all_alphas), 4) if all_alphas else 0.0,
                "mean_bias": round(sum(all_bias) / len(all_bias), 4) if all_bias else 0.0,
                "n_systems": len(sys_agreements),
            }

    result = {"agreements": agreements}

    if output_dir is not None:
        out_path = output_dir / "agreement.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# ─────────────────────────────────── CLI ─────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.v2.eval.agreement",
        description="Compute inter-annotator and inter-judge agreement metrics.",
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "mini", "all"])
    parser.add_argument(
        "--judges",
        nargs="*",
        default=_ALL_JUDGES,
        help="Judges to include in pairwise agreement",
    )
    parser.add_argument("--gold-dir", default=str(_GOLD_DIR))
    parser.add_argument("--cache-dir", default=str(_CACHE_DIR))
    parser.add_argument("--output-dir", default="output/eval")
    args = parser.parse_args(argv)

    from src.v2.eval.runner import _load_manifest, _get_papers, _ALL_SYSTEMS

    # Human annotator agreement
    print("Human annotator agreement:")
    human = compute_human_agreement(Path(args.gold_dir))
    print(f"  n_pairs={human['n_pairs']}  κ={human['cohen_kappa']:.3f}  α={human['krippendorff_alpha']:.3f}")

    # LLM judge agreement
    manifest = _load_manifest()
    papers = _get_papers(manifest, args.split)
    print(f"\nLLM-judge agreement on split={args.split} ({len(papers)} papers):")

    result = compute_all_agreements(
        papers=papers,
        systems=_ALL_SYSTEMS,
        judges=args.judges,
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
    )

    agreements = result["agreements"]
    # Print macro-average per judge pair
    for j_a, j_b in combinations(args.judges, 2):
        pair_key = f"{j_a}_vs_{j_b}"
        if pair_key in agreements:
            ag = agreements[pair_key]
            print(
                f"  {pair_key}: κ={ag['cohen_kappa']:.3f}  α={ag['krippendorff_alpha']:.3f}"
                f"  bias={ag.get('mean_bias', 0.0):+.3f}  (n_systems={ag.get('n_systems', 0)})"
            )

    print(f"\nAgreement results saved to {args.output_dir}/agreement.json")


if __name__ == "__main__":
    main()
