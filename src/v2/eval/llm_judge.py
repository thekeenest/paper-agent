"""
LLM-as-judge cross-model protocol for PaperAffilBench.

Three LLM judges independently score extracted author-affiliation records
against the gold standard, then we compute:
  * Inter-judge Cohen's κ (pairwise) and Krippendorff's α (multi-rater)
  * Per-judge bias toward each system (mean score deviation from ensemble mean)

Judges
------
  claude-sonnet-4-6   (primary)
  gpt-4o              (secondary — requires OPENAI_API_KEY)
  gemini-1.5-flash    (tertiary — requires GOOGLE_API_KEY)

Judgement schema
----------------
Each judge returns per-candidate scores in {0, 1, 2}:
  0 = incorrect (wrong name, wrong affiliation, or hallucinated)
  1 = partial (name correct, affiliation wrong or missing)
  2 = correct (name + affiliation + ROR match gold)

Caching
-------
Judge responses are cached in benchmark/PaperAffilBench/predictions_cache/
as ``<doi_safe>__<judge_model>.json``.  Re-runs read from cache unless
``--force-rerun`` is passed.

Usage
-----
::

    python -m src.v2.eval.llm_judge --split dev --judges claude gpt4o
    python -m src.v2.eval.llm_judge --split test --judges all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import structlog

from src.v2.eval.budget import BudgetExceededError, get_global_tracker

_LOG = structlog.get_logger(__name__)

_MANIFEST_PATH = Path(os.getenv("BENCH_MANIFEST", "benchmark/PaperAffilBench/manifest.json"))
_GOLD_DIR = Path(os.getenv("BENCH_GOLD_DIR", "benchmark/PaperAffilBench/gold"))
_CACHE_DIR = Path(os.getenv("BENCH_CACHE_DIR", "benchmark/PaperAffilBench/predictions_cache"))
_PRED_DIR = Path(os.getenv("BENCH_PRED_DIR", "output/eval"))

_JUDGE_MODELS = {
    "claude": "claude-sonnet-4-6",
    "gpt4o":  "gpt-4o",
    "gemini": "gemini-1.5-flash",
}

_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for scientific author-affiliation extraction.
You will be given:
1. GOLD annotations (ground truth) for a paper
2. PREDICTED annotations from a system

Score each predicted author entry on a 0-2 scale:
  0 = Incorrect: wrong author name, completely wrong affiliation, or hallucinated author
  1 = Partial: author name correct but affiliation wrong or missing
  2 = Correct: author name correct AND normalized affiliation matches AND ROR ID correct (if gold has one)

Return a JSON object:
{
  "scores": [{"author_name": "...", "score": 0|1|2, "reason": "brief reason"}],
  "overall_quality": 0.0-1.0,
  "notes": "any systematic issues observed"
}
"""


# ─────────────────────────── judge result types ──────────────────────────────

class JudgeResult:
    """Result from one judge on one paper."""

    def __init__(
        self,
        doi: str,
        judge: str,
        system: str,
        scores: list[dict[str, Any]],
        overall_quality: float,
        notes: str,
        cached: bool = False,
    ) -> None:
        self.doi = doi
        self.judge = judge
        self.system = system
        self.scores = scores
        self.overall_quality = overall_quality
        self.notes = notes
        self.cached = cached

    def label_sequence(self) -> list[int]:
        """Return integer score sequence for κ/α computation."""
        return [int(s.get("score", 0)) for s in self.scores]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doi": self.doi,
            "judge": self.judge,
            "system": self.system,
            "scores": self.scores,
            "overall_quality": self.overall_quality,
            "notes": self.notes,
            "cached": self.cached,
        }


# ─────────────────────────── calling judges ──────────────────────────────────


def _cache_key(doi: str, judge: str, system: str) -> str:
    safe = doi.replace("/", "_").replace(".", "_")
    return f"{safe}__{judge}__{system}.json"


def _load_from_cache(doi: str, judge: str, system: str) -> dict[str, Any] | None:
    path = _CACHE_DIR / _cache_key(doi, judge, system)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _save_to_cache(doi: str, judge: str, system: str, data: dict[str, Any]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / _cache_key(doi, judge, system)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_judge_prompt(
    gold_annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    paper_title: str = "",
) -> str:
    gold_block = json.dumps(gold_annotations, indent=2)
    pred_block = json.dumps(predictions, indent=2)
    return (
        f"Paper: {paper_title}\n\n"
        f"GOLD annotations:\n{gold_block}\n\n"
        f"PREDICTED annotations:\n{pred_block}\n\n"
        "Score each predicted entry per the instructions."
    )


async def _call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> dict[str, Any]:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    tracker = get_global_tracker()
    # Estimate token counts (rough: 4 chars ≈ 1 token)
    est_in = len(_JUDGE_SYSTEM_PROMPT + prompt) // 4
    tracker.charge(model=model, input_tokens=est_in, output_tokens=300, label="llm_judge")

    llm = ChatAnthropic(model_name=model, temperature=0.0, max_tokens=600)  # type: ignore[call-arg]
    resp = await llm.ainvoke(
        [SystemMessage(content=_JUDGE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    return _parse_judge_response(str(content))


async def _call_openai(prompt: str, model: str = "gpt-4o") -> dict[str, Any]:
    import openai  # type: ignore[import]

    tracker = get_global_tracker()
    est_in = len(_JUDGE_SYSTEM_PROMPT + prompt) // 4
    tracker.charge(model=model, input_tokens=est_in, output_tokens=300, label="llm_judge")

    client = openai.AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return _parse_judge_response(content)


async def _call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> dict[str, Any]:
    try:
        import google.generativeai as genai  # type: ignore[import]

        tracker = get_global_tracker()
        est_in = len(_JUDGE_SYSTEM_PROMPT + prompt) // 4
        tracker.charge(model="gemini-1.5-flash", input_tokens=est_in, output_tokens=300, label="llm_judge")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
        m = genai.GenerativeModel(model)
        resp = await m.generate_content_async(
            f"{_JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
            generation_config={"temperature": 0.0, "max_output_tokens": 600},
        )
        return _parse_judge_response(resp.text)
    except Exception as exc:
        _LOG.warning("judge.gemini_unavailable", error=str(exc))
        return {"scores": [], "overall_quality": 0.0, "notes": f"unavailable: {exc}"}


def _parse_judge_response(content: str) -> dict[str, Any]:
    import re

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"scores": [], "overall_quality": 0.5, "notes": "parse_error"}


async def call_judge(
    judge: str,
    doi: str,
    system: str,
    gold_annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    paper_title: str = "",
    force_rerun: bool = False,
) -> JudgeResult:
    """Call one LLM judge for one (doi, system) pair.

    Uses cache unless force_rerun=True.
    """
    if not force_rerun:
        cached = _load_from_cache(doi, judge, system)
        if cached:
            return JudgeResult(
                doi=doi, judge=judge, system=system,
                scores=cached.get("scores", []),
                overall_quality=cached.get("overall_quality", 0.5),
                notes=cached.get("notes", ""),
                cached=True,
            )

    prompt = _build_judge_prompt(gold_annotations, predictions, paper_title)
    try:
        model = _JUDGE_MODELS.get(judge, judge)
        if judge == "claude":
            result_dict = await _call_claude(prompt, model)
        elif judge == "gpt4o":
            result_dict = await _call_openai(prompt, model)
        elif judge == "gemini":
            result_dict = await _call_gemini(prompt, model)
        else:
            result_dict = {"scores": [], "overall_quality": 0.5, "notes": f"unknown judge {judge}"}
    except BudgetExceededError:
        raise
    except Exception as exc:
        _LOG.warning("judge.call_failed", judge=judge, doi=doi, error=str(exc))
        result_dict = {"scores": [], "overall_quality": 0.0, "notes": f"error: {exc}"}

    jr = JudgeResult(
        doi=doi, judge=judge, system=system,
        scores=result_dict.get("scores", []),
        overall_quality=float(result_dict.get("overall_quality", 0.5)),
        notes=str(result_dict.get("notes", "")),
        cached=False,
    )
    _save_to_cache(doi, judge, system, result_dict)
    return jr


# ─────────────────────────── agreement metrics ───────────────────────────────


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's κ for two raters on aligned ordinal labels (0/1/2)."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))
    # Observed agreement
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    # Expected agreement
    p_e = sum(
        (labels_a.count(k) / n) * (labels_b.count(k) / n)
        for k in categories
    )
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def krippendorff_alpha(ratings: list[list[int]]) -> float:
    """Compute Krippendorff's α for multiple raters (ordinal metric).

    Parameters
    ----------
    ratings:
        List of rater label sequences (each list is one rater's labels).
        All lists must have the same length.
    """
    if not ratings or not ratings[0]:
        return 0.0
    n_raters = len(ratings)
    n_items = len(ratings[0])
    if n_raters < 2:
        return 1.0

    # Observed disagreement (ordinal)
    d_o = 0.0
    n_pairs = 0
    for i in range(n_items):
        item_ratings = [ratings[r][i] for r in range(n_raters)]
        for j in range(n_raters):
            for k in range(j + 1, n_raters):
                d_o += (item_ratings[j] - item_ratings[k]) ** 2
                n_pairs += 1

    if n_pairs == 0:
        return 1.0
    d_o /= n_pairs

    # Expected disagreement
    all_vals = [v for seq in ratings for v in seq]
    mean_v = sum(all_vals) / len(all_vals) if all_vals else 0.0
    var_v = sum((v - mean_v) ** 2 for v in all_vals) / len(all_vals) if all_vals else 0.0
    d_e = 2 * var_v

    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def compute_judge_bias(
    results: list[JudgeResult],
    systems: list[str],
    judges: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-judge bias (mean score - ensemble mean) per system.

    Returns
    -------
    dict[judge → dict[system → bias_score]]
        Positive bias = judge scores this system higher than others on average.
    """
    # Collect mean scores per (judge, system)
    totals: dict[tuple[str, str], list[float]] = {}
    for jr in results:
        key = (jr.judge, jr.system)
        if jr.scores:
            mean_score = sum(s.get("score", 0) for s in jr.scores) / len(jr.scores)
        else:
            mean_score = jr.overall_quality * 2
        totals.setdefault(key, []).append(mean_score)

    mean_scores: dict[tuple[str, str], float] = {
        k: sum(v) / len(v) for k, v in totals.items()
    }

    # Ensemble mean per system (across judges)
    ensemble: dict[str, float] = {}
    for system in systems:
        vals = [mean_scores.get((j, system), 0.0) for j in judges]
        ensemble[system] = sum(vals) / len(vals) if vals else 0.0

    # Bias = judge_mean - ensemble_mean
    bias: dict[str, dict[str, float]] = {j: {} for j in judges}
    for judge in judges:
        for system in systems:
            js = mean_scores.get((judge, system), ensemble[system])
            bias[judge][system] = round(js - ensemble[system], 4)

    return bias


# ─────────────────────────── CLI ─────────────────────────────────────────────


async def run_judge_protocol(
    split: str = "dev",
    judges: list[str] | None = None,
    systems: list[str] | None = None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Run judge protocol and return full results dict."""
    import asyncio

    judges = judges or ["claude"]
    systems = systems or ["full_v2", "plan_act_critic", "openalex_pipeline"]

    if not _MANIFEST_PATH.exists():
        return {"error": "manifest not found"}

    with open(_MANIFEST_PATH) as f:
        manifest = json.load(f)

    split_dois = set(manifest.get("splits", {}).get(split, []))
    papers = {p["doi"]: p for p in manifest["papers"] if p["doi"] in split_dois}

    all_results: list[JudgeResult] = []
    for doi, paper in list(papers.items())[:50]:  # cap at 50 for budget safety
        gold_anns = _load_gold_annotations(doi)
        if not gold_anns:
            continue
        for system in systems:
            preds = _load_system_predictions(doi, system)
            for judge in judges:
                try:
                    jr = await call_judge(
                        judge=judge, doi=doi, system=system,
                        gold_annotations=gold_anns, predictions=preds,
                        paper_title=paper.get("title", ""),
                        force_rerun=force_rerun,
                    )
                    all_results.append(jr)
                except BudgetExceededError as e:
                    _LOG.warning("judge.budget_exceeded", error=str(e))
                    break

    # Compute inter-judge agreement
    agreement: dict[str, Any] = {}
    if len(judges) >= 2:
        for sys in systems:
            sys_results = {j: [] for j in judges}
            for jr in all_results:
                if jr.system == sys:
                    sys_results[jr.judge].extend(jr.label_sequence())

            min_len = min(len(v) for v in sys_results.values()) if sys_results else 0
            if min_len > 0:
                trimmed = {j: v[:min_len] for j, v in sys_results.items()}
                kappa_pairs = {}
                for i, ja in enumerate(judges):
                    for jb in judges[i + 1:]:
                        k = cohen_kappa(trimmed[ja], trimmed[jb])
                        kappa_pairs[f"{ja}_vs_{jb}"] = round(k, 4)

                alpha = krippendorff_alpha(list(trimmed.values()))
                agreement[sys] = {"kappa_pairs": kappa_pairs, "alpha": round(alpha, 4)}

    bias = compute_judge_bias(all_results, systems, judges)

    return {
        "split": split,
        "n_papers": len(papers),
        "n_results": len(all_results),
        "agreement": agreement,
        "bias": bias,
        "results": [jr.to_dict() for jr in all_results],
    }


def _load_gold_annotations(doi: str) -> list[dict[str, Any]]:
    safe = doi.replace("/", "_").replace(".", "_")
    for p in [_GOLD_DIR / f"{safe}.json", _GOLD_DIR / f"{safe}.a.json"]:
        if p.exists():
            return json.loads(p.read_text()).get("annotations", [])
    return []


def _load_system_predictions(doi: str, system: str) -> list[dict[str, Any]]:
    safe = doi.replace("/", "_").replace(".", "_")
    # Look in output/eval/*/predictions/<safe>.json
    import glob

    pattern = str(_PRED_DIR / "*" / "predictions" / f"{safe}.json")
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    for match in matches:
        try:
            data = json.loads(Path(match).read_text())
            if data.get("system") == system:
                return data.get("predictions", [])
        except Exception:
            continue
    return []


def main(argv: list[str] | None = None) -> None:
    import asyncio

    parser = argparse.ArgumentParser(prog="python -m src.v2.eval.llm_judge")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "mini"])
    parser.add_argument("--judges", nargs="+", default=["claude"],
                        choices=["claude", "gpt4o", "gemini", "all"])
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--output", default="output/eval/judge_results.json")
    args = parser.parse_args(argv)

    judges = list(_JUDGE_MODELS.keys()) if "all" in args.judges else args.judges
    results = asyncio.run(run_judge_protocol(
        split=args.split,
        judges=judges,
        systems=args.systems,
        force_rerun=args.force_rerun,
    ))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Judge protocol complete. Results: {out}")
    if "agreement" in results:
        print(json.dumps(results["agreement"], indent=2))
    if "bias" in results:
        print("\nJudge bias:", json.dumps(results["bias"], indent=2))


if __name__ == "__main__":
    main()
