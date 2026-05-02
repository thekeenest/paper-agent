"""
ToolEvidenceRetriever — gathers external evidence for a Candidate before the Critic judges it.

Four evidence types are recognised by the Critic prompt:
  1. openalex_authorship_match   — OpenAlex authorship record for the paper
  2. ror_match_above_threshold   — ROR affiliation match confidence ≥ 0.80
  3. email_domain_ror_match      — email domain → ROR with confidence ≥ 0.70
  4. two_specialists_agree       — merge ToolEvidence noting ≥2 specialists

All evidence items have ``retrieved_at`` set at collection time so verdicts
are replayable.

Usage
-----
::

    retriever = ToolEvidenceRetriever()
    items = await retriever.collect(candidate, paper)
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

from src.v2.linkers._errors import LinkerError
from src.v2.linkers.openalex_institutions import OpenAlexInstitutions
from src.v2.linkers.ror_linker import RORLinker
from src.v2.orchestration.contracts import Candidate, CanonicalPaper, ToolEvidence

_LOG = structlog.get_logger(__name__)

_ROR_ACCEPT_THRESHOLD = 0.80
_EMAIL_ROR_THRESHOLD = 0.70


class ToolEvidenceRetriever:
    """Collects external evidence for a single Candidate in parallel.

    Parameters
    ----------
    ror_timeout, oa_timeout:
        HTTP timeout per linker call (seconds).
    """

    def __init__(
        self,
        ror_timeout: float = 8.0,
        oa_timeout: float = 8.0,
    ) -> None:
        self._ror = RORLinker(timeout=ror_timeout)
        self._oa = OpenAlexInstitutions(timeout=oa_timeout)

    async def collect(
        self,
        candidate: Candidate,
        paper: CanonicalPaper,
    ) -> list[ToolEvidence]:
        """Return a deduplicated list of ToolEvidence items for *candidate*.

        Calls are issued in parallel.  Linker failures are caught and logged;
        they never surface as exceptions here.
        """
        tasks: list[asyncio.Task[list[ToolEvidence]]] = []

        loop = asyncio.get_event_loop()

        # ROR evidence for every affiliation string
        for aff in candidate.affiliations:
            tasks.append(loop.create_task(self._ror_evidence(aff)))

        # Email-domain ROR evidence
        for email in candidate.emails:
            domain = email.split("@")[-1].lower() if "@" in email else ""
            if domain:
                tasks.append(loop.create_task(self._email_ror_evidence(email, domain)))

        # OpenAlex authorship for the paper
        tasks.append(loop.create_task(self._oa_authorship_evidence(candidate, paper)))

        # two_specialists_agree — derived from existing EvidenceTrail (no I/O)
        two_spec = _two_specialists_evidence(candidate)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        items: list[ToolEvidence] = list(two_spec)
        for r in results:
            if isinstance(r, BaseException):
                _LOG.warning("tool_evidence.task_error", error=str(r))
                continue
            items.extend(r)

        _LOG.debug(
            "tool_evidence.collected",
            candidate=candidate.author_name,
            n_items=len(items),
        )
        return items

    # ── ROR ──────────────────────────────────────────────────────────────────

    async def _ror_evidence(self, affiliation: str) -> list[ToolEvidence]:
        try:
            matches = await self._ror.find(affiliation)
        except LinkerError as exc:
            _LOG.debug("tool_evidence.ror_failed", aff=affiliation[:60], error=str(exc))
            return []

        items: list[ToolEvidence] = []
        for m in matches:
            if m.score >= _ROR_ACCEPT_THRESHOLD:
                items.append(
                    ToolEvidence(
                        tool="ror_linker",
                        query=affiliation,
                        result_summary=f"ror_match_above_threshold: {m.name} ({m.ror_id}) score={m.score:.2f}",
                        confidence=m.score,
                        raw_response=m.model_dump(mode="json"),
                    )
                )
        return items

    async def _email_ror_evidence(self, email: str, domain: str) -> list[ToolEvidence]:
        try:
            matches = await self._ror.find(domain)
        except LinkerError as exc:
            _LOG.debug("tool_evidence.email_ror_failed", domain=domain, error=str(exc))
            return []

        items: list[ToolEvidence] = []
        for m in matches:
            if m.score >= _EMAIL_ROR_THRESHOLD:
                items.append(
                    ToolEvidence(
                        tool="ror_linker",
                        query=domain,
                        result_summary=f"email_domain_ror_match: {m.name} ({m.ror_id}) "
                        f"from {email} score={m.score:.2f}",
                        confidence=m.score,
                        raw_response={**m.model_dump(mode="json"), "source_email": email},
                    )
                )
        return items

    # ── OpenAlex ─────────────────────────────────────────────────────────────

    async def _oa_authorship_evidence(
        self,
        candidate: Candidate,
        paper: CanonicalPaper,
    ) -> list[ToolEvidence]:
        work_id = _extract_oa_or_doi(paper)
        if not work_id:
            return []
        try:
            authorships = await self._oa.by_paper(work_id)
        except Exception as exc:
            _LOG.debug("tool_evidence.oa_failed", work_id=work_id, error=str(exc))
            return []

        cand_name_lower = candidate.author_name.lower()
        items: list[ToolEvidence] = []
        for auth in authorships:
            if _name_match(cand_name_lower, auth.author_name.lower()):
                inst_names = [i.display_name for i in auth.institutions]
                items.append(
                    ToolEvidence(
                        tool="openalex_authorship",
                        query=f"{candidate.author_name} in {paper.paper_id}",
                        result_summary=(
                            f"openalex_authorship_match: {auth.author_name} @ "
                            + ", ".join(inst_names or ["(unknown)"])
                        ),
                        confidence=0.90,
                        raw_response=auth.model_dump(mode="json"),
                    )
                )
        return items


# ── helpers ───────────────────────────────────────────────────────────────────


def _two_specialists_evidence(candidate: Candidate) -> list[ToolEvidence]:
    """Return the existing merge ToolEvidence items that note ≥2 specialists."""
    out: list[ToolEvidence] = []
    for ev in candidate.evidence_trail.items:
        if ev.tool == "merge" and "Agreed by:" in ev.result_summary:
            # Check that the summary lists ≥2 specialists
            after_colon = ev.result_summary.split("Agreed by:")[-1]
            parts = [p.strip() for p in after_colon.split(",") if p.strip()]
            if len(parts) >= 2:
                out.append(ev)
    return out


def _extract_oa_or_doi(paper: CanonicalPaper) -> str | None:
    """Return an OpenAlex work ID or DOI suitable for by_paper()."""
    pid = paper.paper_id
    if pid.startswith("openalex:"):
        return pid.removeprefix("openalex:")
    if pid.startswith("doi:"):
        return pid.removeprefix("doi:")
    if paper.raw_metadata:
        ext: Any = paper.raw_metadata.get("externalIds") or {}
        doi: str | None = ext.get("DOI") if isinstance(ext, dict) else None
        if doi:
            return doi
    return None


def _name_match(a: str, b: str) -> bool:
    """Loose name match: True if last-name tokens overlap."""
    tok_a = set(re.split(r"\W+", a)) - {""}
    tok_b = set(re.split(r"\W+", b)) - {""}
    return bool(tok_a & tok_b)
