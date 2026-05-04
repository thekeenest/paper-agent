"""
Evaluation metrics for author-affiliation extraction.

Metrics
-------
  precision / recall / F1     — exact match on normalised affiliation set per paper
  surface_form_match          — token-level F1 between extracted and gold affiliation strings
  ror_linking_accuracy        — fraction of accepted authors with correct ROR ID
  country_accuracy            — fraction of accepted authors with correct country_code
  type_accuracy               — fraction of accepted authors with correct org_type
  all_correct                 — fraction of papers where ALL authors and affiliations correct
  ece                         — Expected Calibration Error from Critic confidences

All metrics operate on a list of (gold, prediction) PaperResult pairs.

Usage
-----
::

    from src.v2.eval.metrics import EvalResult, MetricsCalculator, PaperResult

    results = [PaperResult(gold=gold_dict, prediction=pred_dict) for ...]
    calc = MetricsCalculator(results)
    report = calc.compute_all()
    print(report.to_table())
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────── data types ──────────────────────────────────────


@dataclass
class AuthorGold:
    """Gold-standard annotation for one author."""

    author_name: str
    normalized_aff: str = ""
    ror_id: str = ""
    country_code: str = ""
    org_type: str = ""  # field is "type" in JSON but renamed to avoid built-in clash


@dataclass
class AuthorPred:
    """Predicted author-affiliation record."""

    author_name: str
    normalized_aff: str = ""
    ror_id: str = ""
    country_code: str = ""
    org_type: str = ""
    confidence: float = 1.0


@dataclass
class PaperResult:
    """Gold + prediction for one paper."""

    doi: str
    gold: list[AuthorGold]
    pred: list[AuthorPred]

    @classmethod
    def from_dicts(
        cls,
        doi: str,
        gold_annotations: list[dict[str, Any]],
        pred_candidates: list[dict[str, Any]],
    ) -> "PaperResult":
        gold = [
            AuthorGold(
                author_name=a.get("author_name", ""),
                normalized_aff=a.get("normalized_aff", ""),
                ror_id=a.get("ror_id", ""),
                country_code=a.get("country_code", ""),
                org_type=a.get("type", ""),
            )
            for a in gold_annotations
        ]
        pred = [
            AuthorPred(
                author_name=c.get("author_name", ""),
                normalized_aff=(c.get("affiliations") or [""])[0],
                ror_id=c.get("ror_id", ""),
                country_code=c.get("country_code", ""),
                org_type=c.get("org_type", ""),
                confidence=c.get("confidence", 1.0),
            )
            for c in pred_candidates
        ]
        return cls(doi=doi, gold=gold, pred=pred)


@dataclass
class MetricsReport:
    """Aggregated metrics across all papers in a benchmark run."""

    n_papers: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    surface_form_f1: float = 0.0
    ror_linking_accuracy: float = 0.0
    country_accuracy: float = 0.0
    type_accuracy: float = 0.0
    all_correct: float = 0.0
    ece: float = 0.0
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_papers": self.n_papers,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "surface_form_f1": round(self.surface_form_f1, 4),
            "ror_linking_accuracy": round(self.ror_linking_accuracy, 4),
            "country_accuracy": round(self.country_accuracy, 4),
            "type_accuracy": round(self.type_accuracy, 4),
            "all_correct": round(self.all_correct, 4),
            "ece": round(self.ece, 4),
            "run_id": self.run_id,
        }

    def to_table(self) -> str:
        """Return a human-readable ASCII table."""
        lines = [
            f"{'Metric':<30} {'Value':>8}",
            "-" * 40,
            f"{'n_papers':<30} {self.n_papers:>8d}",
            f"{'precision':<30} {self.precision:>8.4f}",
            f"{'recall':<30} {self.recall:>8.4f}",
            f"{'F1':<30} {self.f1:>8.4f}",
            f"{'surface_form_F1':<30} {self.surface_form_f1:>8.4f}",
            f"{'ror_linking_accuracy':<30} {self.ror_linking_accuracy:>8.4f}",
            f"{'country_accuracy':<30} {self.country_accuracy:>8.4f}",
            f"{'type_accuracy':<30} {self.type_accuracy:>8.4f}",
            f"{'all_correct':<30} {self.all_correct:>8.4f}",
            f"{'ECE':<30} {self.ece:>8.4f}",
        ]
        if self.run_id:
            lines.append(f"{'run_id':<30} {self.run_id:>8s}")
        return "\n".join(lines)

    def to_booktabs_row(self, label: str) -> str:
        """Return one LaTeX booktabs table row for this result."""
        return (
            f"{label} & {self.precision:.3f} & {self.recall:.3f} & {self.f1:.3f}"
            f" & {self.ror_linking_accuracy:.3f} & {self.country_accuracy:.3f}"
            f" & {self.type_accuracy:.3f} & {self.all_correct:.3f}"
            f" & {self.ece:.3f} \\\\"
        )


# ─────────────────────────── calculator ──────────────────────────────────────


class MetricsCalculator:
    """Compute all metrics from a list of PaperResult objects.

    Parameters
    ----------
    results:
        One entry per paper.
    """

    def __init__(self, results: list[PaperResult]) -> None:
        self._results = results

    def compute_all(self, run_id: str = "") -> MetricsReport:
        """Compute and aggregate all metrics."""
        if not self._results:
            return MetricsReport(run_id=run_id)

        prec_vals: list[float] = []
        rec_vals: list[float] = []
        f1_vals: list[float] = []
        sf_f1_vals: list[float] = []
        ror_acc_vals: list[float] = []
        country_acc_vals: list[float] = []
        type_acc_vals: list[float] = []
        all_correct_vals: list[float] = []
        calib_pairs: list[tuple[float, bool]] = []

        for pr in self._results:
            p, r, f1 = self._prf1(pr)
            prec_vals.append(p)
            rec_vals.append(r)
            f1_vals.append(f1)
            sf_f1_vals.append(self._surface_form_f1(pr))
            ror_acc_vals.append(self._ror_accuracy(pr))
            country_acc_vals.append(self._country_accuracy(pr))
            type_acc_vals.append(self._type_accuracy(pr))
            all_correct_vals.append(1.0 if self._all_correct(pr) else 0.0)
            calib_pairs.extend(self._calibration_pairs(pr))

        def avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        report = MetricsReport(
            n_papers=len(self._results),
            precision=avg(prec_vals),
            recall=avg(rec_vals),
            f1=avg(f1_vals),
            surface_form_f1=avg(sf_f1_vals),
            ror_linking_accuracy=avg(ror_acc_vals),
            country_accuracy=avg(country_acc_vals),
            type_accuracy=avg(type_acc_vals),
            all_correct=avg(all_correct_vals),
            ece=_compute_ece(calib_pairs),
            run_id=run_id,
        )
        return report

    # ── per-paper metrics ──────────────────────────────────────────────────

    def _prf1(self, pr: PaperResult) -> tuple[float, float, float]:
        """Author-level P/R/F1 based on exact author_name + normalized_aff match."""
        gold_set = {
            (_norm_name(a.author_name), _norm_aff(a.normalized_aff))
            for a in pr.gold
        }
        pred_set = {
            (_norm_name(a.author_name), _norm_aff(a.normalized_aff))
            for a in pr.pred
        }
        if not gold_set and not pred_set:
            return 1.0, 1.0, 1.0
        if not gold_set:
            return 0.0, 1.0, 0.0
        if not pred_set:
            return 1.0, 0.0, 0.0
        tp = len(gold_set & pred_set)
        p = tp / len(pred_set)
        r = tp / len(gold_set)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def _surface_form_f1(self, pr: PaperResult) -> float:
        """Token-level F1 comparing affiliation strings."""
        gold_tokens = _tokenise(" ".join(a.normalized_aff for a in pr.gold))
        pred_tokens = _tokenise(" ".join(a.normalized_aff for a in pr.pred))
        return _token_f1(gold_tokens, pred_tokens)

    def _ror_accuracy(self, pr: PaperResult) -> float:
        """Fraction of gold authors whose ROR ID is correctly predicted."""
        if not pr.gold:
            return 1.0
        gold_ror = {_norm_name(a.author_name): a.ror_id for a in pr.gold if a.ror_id}
        if not gold_ror:
            return 1.0  # no ROR to judge
        pred_ror = {_norm_name(a.author_name): a.ror_id for a in pr.pred}
        n_correct = sum(
            1 for name, ror in gold_ror.items() if pred_ror.get(name) == ror
        )
        return n_correct / len(gold_ror)

    def _country_accuracy(self, pr: PaperResult) -> float:
        """Fraction of gold authors with correct country_code."""
        gold_with_country = [a for a in pr.gold if a.country_code]
        if not gold_with_country:
            return 1.0
        pred_country = {_norm_name(a.author_name): a.country_code.upper() for a in pr.pred}
        n_correct = sum(
            1 for a in gold_with_country
            if pred_country.get(_norm_name(a.author_name)) == a.country_code.upper()
        )
        return n_correct / len(gold_with_country)

    def _type_accuracy(self, pr: PaperResult) -> float:
        """Fraction of gold authors with correct org_type (education/company/etc.)."""
        gold_with_type = [a for a in pr.gold if a.org_type]
        if not gold_with_type:
            return 1.0
        pred_type = {_norm_name(a.author_name): a.org_type for a in pr.pred}
        n_correct = sum(
            1 for a in gold_with_type
            if pred_type.get(_norm_name(a.author_name)) == a.org_type
        )
        return n_correct / len(gold_with_type)

    def _all_correct(self, pr: PaperResult) -> bool:
        """True iff P/R/F1 == 1.0 AND all ROR, country, type correct."""
        p, r, f1 = self._prf1(pr)
        return (
            f1 == 1.0
            and self._ror_accuracy(pr) == 1.0
            and self._country_accuracy(pr) == 1.0
            and self._type_accuracy(pr) == 1.0
        )

    def _calibration_pairs(
        self, pr: PaperResult
    ) -> list[tuple[float, bool]]:
        """Return (confidence, is_correct) pairs for Critic calibration."""
        gold_set = {_norm_name(a.author_name) for a in pr.gold}
        pairs = []
        for a in pr.pred:
            is_correct = _norm_name(a.author_name) in gold_set
            pairs.append((a.confidence, is_correct))
        return pairs


# ─────────────────────────── helpers ─────────────────────────────────────────


def _norm_name(name: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _norm_aff(aff: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    s = re.sub(r"[^\w\s]", " ", aff.lower())
    return re.sub(r"\s+", " ", s).strip()


def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _token_f1(gold_tokens: list[str], pred_tokens: list[str]) -> float:
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    gold_count: dict[str, int] = {}
    for t in gold_tokens:
        gold_count[t] = gold_count.get(t, 0) + 1
    pred_count: dict[str, int] = {}
    for t in pred_tokens:
        pred_count[t] = pred_count.get(t, 0) + 1
    overlap = sum(
        min(gold_count.get(t, 0), pred_count.get(t, 0)) for t in set(gold_count) | set(pred_count)
    )
    if overlap == 0:
        return 0.0
    p = overlap / sum(pred_count.values())
    r = overlap / sum(gold_count.values())
    return 2 * p * r / (p + r)


def _compute_ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float:
    """Expected Calibration Error over (confidence, is_correct) pairs."""
    if not pairs:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))
    ece = 0.0
    n_total = len(pairs)
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / n_total) * abs(avg_conf - avg_acc)
    return ece
