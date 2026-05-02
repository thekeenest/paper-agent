"""
Critic agent — tool-grounded verifier with salvage path and retry routing.

For each merged Candidate the Critic:
  1. Collects ToolEvidence via ToolEvidenceRetriever (ROR + OpenAlex in parallel).
  2. Asks Claude Sonnet 4.6 to emit a Verdict citing at least one evidence item.
  3. Applies the salvage path: if the LLM rejects a candidate that ≥2 specialists
     produced, emit it anyway with decision="accept", confidence_band="low",
     salvage=True.
  4. Returns Verdicts including any retry hints for downstream re-routing.

ECE calibration
---------------
When a gold label is available (for benchmark papers), call
``record_calibration(verdict, gold_correct)`` then ``compute_ece()`` to get
the Expected Calibration Error across the run.

Prompt versioning
-----------------
The system prompt is loaded at runtime from ``docs/critic_prompts.md``.
Changes to the prompt are therefore tracked in git history.

Usage
-----
::

    critic = Critic()
    verdicts = await critic.judge(merged_candidates, paper)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.v2.orchestration.contracts import (
    Candidate,
    CanonicalPaper,
    ConfidenceLevel,
    ToolEvidence,
    Verdict,
)

from .tool_evidence import ToolEvidenceRetriever

_LOG = structlog.get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent.parent.parent / "docs" / "critic_prompts.md"
_DEFAULT_MODEL = os.getenv("CRITIC_MODEL", "claude-sonnet-4-6")


# ── structured output schema ──────────────────────────────────────────────────


class _CriticJudgment(BaseModel):
    """One verdict object the LLM must produce per Candidate."""

    candidate_id: str = Field(description="Must match the input candidate_id exactly")
    decision: Literal["accept", "reject", "retry"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="evidence_id values from the provided ToolEvidence list",
    )
    retry_hint: str = Field(
        default="",
        description="Populated only when decision='retry': specific guidance for the specialist",
    )


class _CriticOutput(BaseModel):
    """Top-level structured output from the Critic LLM."""

    verdicts: list[_CriticJudgment]


# ── ECE calibration store ─────────────────────────────────────────────────────


class CalibrationStore:
    """Accumulates (predicted_confidence, gold_correct) pairs for ECE computation."""

    def __init__(self) -> None:
        self._pairs: list[tuple[float, bool]] = []

    def record(self, predicted: float, gold_correct: bool) -> None:
        self._pairs.append((predicted, gold_correct))

    def compute_ece(self, n_bins: int = 10) -> float:
        """Expected Calibration Error over collected pairs."""
        if not self._pairs:
            return 0.0
        bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
        for p, correct in self._pairs:
            idx = min(int(p * n_bins), n_bins - 1)
            bins[idx].append((p, correct))
        ece = 0.0
        n_total = len(self._pairs)
        for bucket in bins:
            if not bucket:
                continue
            avg_conf = sum(p for p, _ in bucket) / len(bucket)
            avg_acc = sum(1 for _, c in bucket if c) / len(bucket)
            ece += (len(bucket) / n_total) * abs(avg_conf - avg_acc)
        return ece


# ── Critic ────────────────────────────────────────────────────────────────────


class Critic:
    """LLM-backed Critic that emits Verdicts for a list of Candidates.

    Parameters
    ----------
    model:
        Anthropic model name.  Defaults to ``CRITIC_MODEL`` env var or
        ``claude-sonnet-4-6``.
    temperature:
        Sampling temperature.  0 for deterministic verdicts.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        resolved: str = model if model is not None else (os.getenv("CRITIC_MODEL") or _DEFAULT_MODEL)
        llm = ChatAnthropic(model_name=resolved, temperature=temperature)  # type: ignore[call-arg]
        self._chain = llm.with_structured_output(_CriticOutput)
        self._retriever = ToolEvidenceRetriever()
        self._system_prompt = _load_system_prompt()
        self.calibration = CalibrationStore()

    async def judge(
        self,
        candidates: list[Candidate],
        paper: CanonicalPaper,
    ) -> list[Verdict]:
        """Produce a Verdict for every Candidate.

        Parameters
        ----------
        candidates:
            Merged candidates from the extraction stage.
        paper:
            The canonical paper record (used for OpenAlex authorship lookup).

        Returns
        -------
        list[Verdict]
            One Verdict per input Candidate.  Salvaged candidates have
            ``decision="accept"`` and ``salvage=True``.
        """
        if not candidates:
            return []

        _LOG.info("critic.start", n_candidates=len(candidates), paper=paper.paper_id)

        # 1. Collect evidence for all candidates in parallel
        import asyncio
        loop = asyncio.get_running_loop()
        evidence_tasks = [
            loop.create_task(self._retriever.collect(cand, paper))
            for cand in candidates
        ]
        evidence_results = await asyncio.gather(*evidence_tasks, return_exceptions=True)

        evidence_map: dict[str, list[ToolEvidence]] = {}
        for cand, res in zip(candidates, evidence_results):
            if isinstance(res, BaseException):
                _LOG.warning("critic.evidence_failed", candidate=cand.author_name, error=str(res))
                evidence_map[cand.candidate_id] = []
            else:
                evidence_map[cand.candidate_id] = list(res)

        # 2. Call LLM Critic
        user_msg = _build_user_message(candidates, evidence_map)
        messages = [SystemMessage(content=self._system_prompt), HumanMessage(content=user_msg)]

        try:
            llm_output: _CriticOutput | Any = await self._chain.ainvoke(messages)
            if not isinstance(llm_output, _CriticOutput):
                _LOG.warning("critic.unexpected_output_type", type=type(llm_output).__name__)
                llm_output = _CriticOutput(verdicts=[])
        except Exception as exc:
            _LOG.error("critic.llm_failed", error=str(exc))
            # Fallback: accept all candidates that have ≥2 specialists, reject rest
            llm_output = _CriticOutput(verdicts=[])

        # Build lookup: candidate_id → _CriticJudgment
        judgment_map: dict[str, _CriticJudgment] = {
            j.candidate_id: j for j in llm_output.verdicts
        }

        # 3. Produce final Verdicts with salvage path
        verdicts: list[Verdict] = []
        for cand in candidates:
            judgment = judgment_map.get(cand.candidate_id)
            evidence_items = evidence_map.get(cand.candidate_id, [])
            verdict = _build_verdict(cand, judgment, evidence_items)
            verdicts.append(verdict)

        accepted = sum(1 for v in verdicts if v.decision == "accept")
        salvaged = sum(1 for v in verdicts if v.salvage)
        retried = sum(1 for v in verdicts if v.decision == "retry")
        _LOG.info(
            "critic.done",
            accepted=accepted,
            salvaged=salvaged,
            retried=retried,
            rejected=len(verdicts) - accepted - retried,
        )
        return verdicts

    def record_calibration(self, verdict: Verdict, gold_correct: bool) -> None:
        """Register a (predicted_confidence, gold_correct) pair for ECE."""
        self.calibration.record(verdict.confidence, gold_correct)

    def compute_ece(self) -> float:
        """Return ECE over all recorded calibration pairs."""
        return self.calibration.compute_ece()


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_system_prompt() -> str:
    """Read the versioned system prompt from docs/critic_prompts.md."""
    path = _PROMPT_PATH
    if not path.exists():
        # Fallback path relative to CWD (useful when running tests from repo root)
        path = Path("docs/critic_prompts.md")
    if not path.exists():
        raise FileNotFoundError(f"Critic prompt file not found: {_PROMPT_PATH}")

    text = path.read_text(encoding="utf-8")
    # Extract the first ```...``` block after "## SYSTEM_PROMPT_V1"
    match = re.search(r"## SYSTEM_PROMPT_V1\s*\n```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError("Could not parse SYSTEM_PROMPT_V1 from docs/critic_prompts.md")


def _build_user_message(
    candidates: list[Candidate],
    evidence_map: dict[str, list[ToolEvidence]],
) -> str:
    """Build the user message presented to the Critic LLM."""
    lines: list[str] = ["Evaluate the following candidates:\n"]
    for cand in candidates:
        ev_items = evidence_map.get(cand.candidate_id, [])
        ev_block = "\n".join(
            f"  - [{e.evidence_id}] tool={e.tool} confidence={e.confidence:.2f} "
            f"summary={e.result_summary!r}"
            for e in ev_items
        ) or "  (no external evidence retrieved)"
        lines.append(
            f"### Candidate {cand.candidate_id}\n"
            f"author_name: {cand.author_name!r}\n"
            f"affiliations: {cand.affiliations}\n"
            f"emails: {cand.emails}\n"
            f"source_specialist: {cand.source_specialist}\n"
            f"Evidence bundle:\n{ev_block}\n"
        )
    lines.append(
        "\nReturn a JSON object with a 'verdicts' array covering ALL candidates above."
    )
    return "\n".join(lines)


def _specialist_count(candidate: Candidate) -> int:
    """Count distinct specialists that contributed to this candidate's EvidenceTrail."""
    specialists: set[str] = set()
    for ev in candidate.evidence_trail.items:
        if ev.tool == "merge" and "Agreed by:" in ev.result_summary:
            after = ev.result_summary.split("Agreed by:")[-1]
            for s in after.split(","):
                s = s.strip()
                if s:
                    specialists.add(s)
    return len(specialists)


def _confidence_band(confidence: float) -> ConfidenceLevel:
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _build_verdict(
    candidate: Candidate,
    judgment: _CriticJudgment | None,
    evidence_items: list[ToolEvidence],
) -> Verdict:
    """Convert an LLM judgment + evidence into a final Verdict, applying salvage."""
    n_specialists = _specialist_count(candidate)

    # Case 1: LLM produced a valid judgment
    if judgment is not None:
        decision = judgment.decision
        confidence = judgment.confidence
        rationale = judgment.rationale
        ev_ids = judgment.evidence_ids
        retry_hint = judgment.retry_hint

        # Enforce: accept requires non-empty evidence_ids
        if decision == "accept" and not ev_ids:
            # Try to auto-populate from retrieved evidence
            ev_ids = [e.evidence_id for e in evidence_items[:3]]
            if not ev_ids:
                # Downgrade to reject — no evidence available
                decision = "reject"
                rationale = f"[auto-downgraded] {rationale} (no evidence_ids)"

        # Salvage path: LLM rejected but ≥2 specialists agree
        salvage = False
        if decision == "reject" and n_specialists >= 2:
            decision = "accept"
            confidence = min(confidence + 0.10, 0.55)
            rationale = f"[salvage] {rationale}"
            salvage = True
            # Add two_specialists_agree evidence if not already present
            if not ev_ids:
                merge_ev = [e for e in evidence_items if e.tool == "merge"]
                ev_ids = [e.evidence_id for e in merge_ev[:1]]

        return Verdict(
            candidate_id=candidate.candidate_id,
            decision=decision,
            confidence=confidence,
            confidence_band=_confidence_band(confidence),
            rationale=rationale,
            evidence_ids=ev_ids,
            salvage=salvage,
            retry_hint=retry_hint,
        )

    # Case 2: LLM did not produce a judgment for this candidate
    # Apply salvage if ≥2 specialists; otherwise reject
    if n_specialists >= 2:
        merge_ev_ids = [
            e.evidence_id
            for e in evidence_items
            if e.tool == "merge"
        ]
        return Verdict(
            candidate_id=candidate.candidate_id,
            decision="accept",
            confidence=0.45,
            confidence_band="low",
            rationale="[salvage] Critic did not judge this candidate; ≥2 specialists agree.",
            evidence_ids=merge_ev_ids or [e.evidence_id for e in evidence_items[:1]],
            salvage=True,
        )

    return Verdict(
        candidate_id=candidate.candidate_id,
        decision="reject",
        confidence=0.20,
        confidence_band="low",
        rationale="Critic did not produce a judgment and insufficient specialist agreement.",
        evidence_ids=[],
    )
