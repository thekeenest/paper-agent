"""
Specialist merge — combines outputs from header, footnote, acknowledgements,
and email-domain extractors into a deduplicated list of Candidates.

Merge rules
-----------
* Authors appearing in ≥2 specialists → confidence="high"
* Authors appearing in only 1 specialist → keep their per-specialist confidence
* Deduplication key: normalised author name (lowercase, collapse whitespace)
* Affiliations are unioned; emails are unioned
* EvidenceTrail accumulates ToolEvidence from all contributing specialists
* Candidates that pass through are wrapped with which specialists contributed

Usage
-----
::

    from src.v2.agents.extractors.merge import merge_candidates

    merged = merge_candidates(
        header=header_cand_dicts,
        footnote=footnote_cand_dicts,
        ack=ack_cand_dicts,
        email=email_cand_dicts,
    )
"""
from __future__ import annotations

import contextlib
import re
from typing import Any

import structlog

from src.v2.orchestration.contracts import Candidate, ConfidenceLevel, EvidenceTrail, ToolEvidence

_LOG = structlog.get_logger(__name__)


def _norm_name(name: str) -> str:
    """Normalise author name for deduplication key."""
    return re.sub(r"\s+", " ", name.lower().strip())


def merge_candidates(
    header: list[dict[str, Any]],
    footnote: list[dict[str, Any]],
    ack: list[dict[str, Any]],
    email: list[dict[str, Any]],
) -> list[Candidate]:
    """Merge specialist candidate dicts into a deduplicated Candidate list.

    Parameters
    ----------
    header, footnote, ack, email:
        Lists of Candidate model_dump() dicts from the respective specialists.

    Returns
    -------
    list[Candidate]
        Merged, deduplicated candidates with updated confidence and EvidenceTrails.
    """
    # Group by normalised name
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    for specialist, cands in [
        ("header", header),
        ("footnote", footnote),
        ("acknowledgements", ack),
        ("email_domain", email),
    ]:
        for c in cands:
            key = _norm_name(c.get("author_name", ""))
            if not key:
                continue
            if key not in groups:
                groups[key] = []
            groups[key].append((specialist, c))

    merged: list[Candidate] = []
    for _key, entries in groups.items():
        specialists = list({spec for spec, _ in entries})
        all_affiliations: list[str] = []
        all_emails: list[str] = []
        all_span_ids: list[str] = []
        all_evidence_items: list[ToolEvidence] = []

        # Pick canonical name = longest / most-complete version
        best_name = max((c.get("author_name", "") for _, c in entries), key=len)

        for _spec, c in entries:
            for aff in c.get("affiliations", []):
                if aff and aff not in all_affiliations:
                    all_affiliations.append(aff)
            for em in c.get("emails", []):
                if em and em not in all_emails:
                    all_emails.append(em)
            for sid in c.get("evidence_span_ids", []):
                if sid not in all_span_ids:
                    all_span_ids.append(sid)
            # Reconstruct EvidenceTrail items from dict
            trail_data = c.get("evidence_trail", {})
            for ev in trail_data.get("items", []):
                with contextlib.suppress(Exception):
                    all_evidence_items.append(ToolEvidence.model_validate(ev))

        # Add a synthetic merge evidence record
        if len(specialists) >= 2:
            all_evidence_items.append(
                ToolEvidence(
                    tool="merge",
                    query=best_name,
                    result_summary=f"Agreed by: {', '.join(sorted(specialists))}",
                    confidence=1.0,
                )
            )

        confidence: ConfidenceLevel = "high" if len(specialists) >= 2 else "medium"

        try:
            merged.append(
                Candidate(
                    author_name=best_name,
                    affiliations=all_affiliations,
                    emails=all_emails,
                    source_specialist="merge",
                    evidence_span_ids=all_span_ids,
                    confidence=confidence,
                    evidence_trail=EvidenceTrail(items=all_evidence_items),
                )
            )
        except ValueError as exc:
            _LOG.warning("merge.candidate_rejected", name=best_name, reason=str(exc))

    _LOG.info(
        "merge.done",
        n_input=len(header) + len(footnote) + len(ack) + len(email),
        n_merged=len(merged),
        n_high=sum(1 for c in merged if c.confidence == "high"),
    )
    return merged
