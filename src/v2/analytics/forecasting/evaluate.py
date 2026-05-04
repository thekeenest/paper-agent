"""
Forecasting evaluation script.

Trains all three models (DHGNNLite, StaticGCN, CoFreq) and evaluates them
on the 1-year horizon link prediction task, stratified by industry-academia mix.

Output
------
  experiments/forecasting/checkpoints/best.pt  — best DHGNNLite weights
  experiments/forecasting/results.json         — AUC + AP per model

Constraint checked
------------------
  DHGNN-lite AUC ≥ co_freq AUC + 0.05

Usage
-----
::

    python -m src.v2.analytics.forecasting.evaluate \\
        --kg-db output/v2/kg \\
        --epochs 50 \\
        --min-year 2019 --max-year 2023
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from src.v2.analytics.forecasting.co_freq import CoFreqBaseline
from src.v2.analytics.forecasting.dataset import CollabForecastDataset
from src.v2.analytics.forecasting.dhgnn_lite import (
    DHGNNLite,
    evaluate_dhgnn,
    load_checkpoint,
    train_dhgnn,
)
from src.v2.analytics.forecasting.static_gcn import (
    StaticGCN,
    evaluate_static_gcn,
    train_static_gcn,
)

_CHECKPOINT_DIR = Path(os.getenv("FORECAST_CKPT", "experiments/forecasting/checkpoints"))
_RESULTS_DIR = Path(os.getenv("FORECAST_RESULTS", "experiments/forecasting"))
_SEED = int(os.getenv("FORECAST_SEED", "42"))


# ─────────────────────────── stratification ──────────────────────────────────


def _stratify_by_industry_mix(
    test_snapshots: list[Any],
) -> dict[str, list[Any]]:
    """Split test snapshots by industry-academia mix level.

    Returns dict with keys:
      "pure_academia"  — no company nodes in snapshot
      "mixed"          — some company nodes
      "high_industry"  — majority company nodes (>50%)
    """
    groups: dict[str, list[Any]] = {"pure_academia": [], "mixed": [], "high_industry": []}
    for snap in test_snapshots:
        x_inst = snap.data["institution"].x
        if x_inst.shape[0] == 0:
            groups["pure_academia"].append(snap)
            continue
        company_col = 1  # org_type one-hot index for "company"
        n_company = int((x_inst[:, company_col] > 0).sum().item())
        n_total = x_inst.shape[0]
        ratio = n_company / n_total
        if ratio == 0:
            groups["pure_academia"].append(snap)
        elif ratio > 0.5:
            groups["high_industry"].append(snap)
        else:
            groups["mixed"].append(snap)
    return groups


def _evaluate_stratified(
    model_name: str,
    predict_fn,  # callable(test_snaps, train_snaps) -> (auc, ap)
    test_snapshots: list[Any],
    train_snapshots: list[Any],
) -> dict[str, Any]:
    """Evaluate model stratified by industry mix."""
    groups = _stratify_by_industry_mix(test_snapshots)
    results: dict[str, Any] = {}
    for stratum, snaps in groups.items():
        if snaps:
            auc, ap = predict_fn(snaps, train_snapshots)
        else:
            auc, ap = 0.0, 0.0
        results[stratum] = {"auc": round(auc, 4), "ap": round(ap, 4), "n": len(snaps)}
    # Overall
    auc_all, ap_all = predict_fn(test_snapshots, train_snapshots)
    results["overall"] = {"auc": round(auc_all, 4), "ap": round(ap_all, 4), "n": len(test_snapshots)}
    return results


# ─────────────────────────── main evaluation ─────────────────────────────────


def run_evaluation(
    kg_db_path: str = "output/v2/kg",
    min_year: int = 2019,
    max_year: int = 2023,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 64,
    n_layers: int = 2,
    seed: int = _SEED,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full evaluation: train all models, evaluate, save checkpoint.

    Returns
    -------
    dict with model → stratified AUC/AP results.
    """
    torch.manual_seed(seed)

    # Build dataset
    if verbose:
        print(f"Building dataset from KG at {kg_db_path}...")
    ds = CollabForecastDataset(
        kg_db_path=kg_db_path,
        min_year=min_year,
        max_year=max_year,
        seed=seed,
    )
    train_snaps, test_snaps = ds.split()

    if verbose:
        print(f"  Train snapshots: {len(train_snaps)}, test snapshots: {len(test_snaps)}")

    results: dict[str, Any] = {}

    # ── Co-freq baseline ─────────────────────────────────────────────────────
    if verbose:
        print("\n[1/3] Co-frequency baseline...")
    co_freq = CoFreqBaseline()

    def _co_freq_eval(test, train):
        return co_freq.evaluate(test, train)

    results["co_freq"] = _evaluate_stratified("co_freq", _co_freq_eval, test_snaps, train_snaps)
    co_freq_auc = results["co_freq"]["overall"]["auc"]
    if verbose:
        print(f"  Co-freq AUC={co_freq_auc:.4f}")

    # ── Static GCN baseline ──────────────────────────────────────────────────
    if verbose:
        print("\n[2/3] Static GCN baseline...")
    gcn = StaticGCN(in_channels=6, hidden=hidden)
    train_static_gcn(gcn, train_snaps, epochs=epochs, lr=lr, seed=seed, verbose=verbose)

    def _gcn_eval(test, train):
        return evaluate_static_gcn(gcn, test, train)

    results["static_gcn"] = _evaluate_stratified("static_gcn", _gcn_eval, test_snaps, train_snaps)
    gcn_auc = results["static_gcn"]["overall"]["auc"]
    if verbose:
        print(f"  StaticGCN AUC={gcn_auc:.4f}")

    # ── DHGNNLite ─────────────────────────────────────────────────────────────
    if verbose:
        print("\n[3/3] DHGNN-Lite...")
    ckpt_path = _CHECKPOINT_DIR / "best.pt"
    dhgnn = DHGNNLite(author_in=1, inst_in=6, hidden=hidden, n_layers=n_layers)
    if verbose:
        print(f"  Model params: {dhgnn.n_params():,} (limit: 5M)")
    assert dhgnn.n_params() <= 5_000_000, f"DHGNNLite exceeds 5M params: {dhgnn.n_params()}"

    train_dhgnn(dhgnn, train_snaps, epochs=epochs, lr=lr, seed=seed, checkpoint_path=ckpt_path, verbose=verbose)

    # Load best checkpoint if saved
    if ckpt_path.exists():
        dhgnn = load_checkpoint(DHGNNLite(author_in=1, inst_in=6, hidden=hidden, n_layers=n_layers), ckpt_path)

    def _dhgnn_eval(test, train):
        return evaluate_dhgnn(dhgnn, test, train)

    results["dhgnn_lite"] = _evaluate_stratified("dhgnn_lite", _dhgnn_eval, test_snaps, train_snaps)
    dhgnn_auc = results["dhgnn_lite"]["overall"]["auc"]
    if verbose:
        print(f"  DHGNNLite AUC={dhgnn_auc:.4f}")

    # ── Constraint check ─────────────────────────────────────────────────────
    auc_margin = dhgnn_auc - co_freq_auc
    results["constraint_check"] = {
        "dhgnn_auc": dhgnn_auc,
        "co_freq_auc": co_freq_auc,
        "margin": round(auc_margin, 4),
        "passed": auc_margin >= 0.05,
    }
    if verbose:
        status = "PASSED" if auc_margin >= 0.05 else "WARNING (not met)"
        print(
            f"\nConstraint: DHGNN AUC ({dhgnn_auc:.4f}) >= co_freq ({co_freq_auc:.4f}) + 0.05 "
            f"→ margin={auc_margin:.4f} [{status}]"
        )

    # Save results
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if verbose:
        print(f"\nResults written to {out_path}")
        if ckpt_path.exists():
            print(f"Best checkpoint: {ckpt_path}")

    return results


# ─────────────────────────── CLI ─────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.v2.analytics.forecasting.evaluate",
        description="Train and evaluate forecasting models on the KG.",
    )
    parser.add_argument("--kg-db", default="output/v2/kg")
    parser.add_argument("--min-year", type=int, default=2019)
    parser.add_argument("--max-year", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run_evaluation(
        kg_db_path=args.kg_db,
        min_year=args.min_year,
        max_year=args.max_year,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        n_layers=args.n_layers,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
