"""
Email-domain specialist extractor.

Pipeline per email found in ParsedDoc.emails:
  1. Parse domain from address.
  2. DNS MX/A lookup to verify the domain is reachable.
  3. Call RORLinker.find(domain) to get a candidate institution.
  4. Emit a Candidate with provenance="email_domain" and ToolEvidence.

This extractor does NOT call an LLM — it is purely deterministic.
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import structlog

from src.v2.linkers._errors import LinkerError
from src.v2.linkers.ror_linker import RORLinker
from src.v2.orchestration.contracts import (
    Candidate,
    Candidates,
    EvidenceTrail,
    ToolEvidence,
)
from src.v2.parsers.schemas import ParsedDoc

_LOG = structlog.get_logger(__name__)

_INSTITUTION_DOMAIN_RE = re.compile(
    r"\.(edu|ac\.\w{2}|university\.\w+|uni-\w+|tu-\w+|eth|epfl|mit|caltech|stanford|cam\.ac\.uk)$",
    re.I,
)


class EmailDomainExtractor:
    """Non-LLM extractor: email → domain → ROR institution."""

    def __init__(
        self,
        dns_timeout: float = 3.0,
        ror_cache_ttl: float = 7 * 86_400,
    ) -> None:
        self._dns_timeout = dns_timeout
        self._ror = RORLinker(cache_ttl=ror_cache_ttl)

    async def extract(self, doc: ParsedDoc) -> Candidates:
        """Extract Candidates from all emails in *doc*."""
        if not doc.emails:
            return Candidates(items=[], specialist="email_domain")

        tasks = [self._process_email(e.address, e.author_hint) for e in doc.emails]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[Candidate] = []
        for r in results:
            if isinstance(r, BaseException):
                _LOG.warning("email_domain.error", error=str(r))
                continue
            if r is not None:
                items.append(r)

        _LOG.info("email_domain.done", n_emails=len(doc.emails), n_candidates=len(items))
        return Candidates(items=items, specialist="email_domain")

    async def _process_email(
        self,
        address: str,
        author_hint: str | None,
    ) -> Candidate | None:
        domain = address.split("@")[-1].lower() if "@" in address else address
        if not domain:
            return None

        # DNS check (best-effort; don't fail if dnspython not installed)
        domain_valid = await self._dns_check(domain)
        if not domain_valid:
            _LOG.debug("email_domain.dns_miss", domain=domain)
            # Still proceed — ROR might know this domain

        # ROR lookup keyed on domain
        try:
            matches = await self._ror.find(domain)
        except LinkerError as exc:
            _LOG.warning("email_domain.ror_failed", domain=domain, error=str(exc))
            matches = []

        if not matches:
            return None

        best = matches[0]
        evidence = ToolEvidence(
            tool="ror_linker",
            query=domain,
            result_summary=f"{best.name} ({best.ror_id})",
            confidence=best.score,
            raw_response=best.model_dump(mode="json"),
        )
        trail = EvidenceTrail(items=[evidence])

        # Use author_hint if available; otherwise derive from email local part
        name = author_hint or _name_from_email(address)
        if not name or len(name.strip()) < 2:
            return None

        return Candidate(
            author_name=name,
            affiliations=[best.name],
            emails=[address],
            source_specialist="email_domain",
            evidence_span_ids=[],
            confidence="medium",
            evidence_trail=trail,
        )

    async def _dns_check(self, domain: str) -> bool:
        """Return True if domain has a reachable MX or A record."""
        try:
            import dns.asyncresolver
            resolver = dns.asyncresolver.Resolver()
            resolver.timeout = self._dns_timeout
            resolver.lifetime = self._dns_timeout
            try:
                await resolver.resolve(domain, "MX")
                return True
            except Exception:
                await resolver.resolve(domain, "A")
                return True
        except ImportError:
            _LOG.debug("email_domain.dnspython_not_available")
            return True  # assume valid if we can't check
        except Exception:
            return False


def _name_from_email(address: str) -> str:
    """Best-effort name extraction from email local part.

    e.g. "alice.smith@mit.edu" → "Alice Smith"
         "asmith@example.com" → "" (too ambiguous)
    """
    local = address.split("@")[0] if "@" in address else ""
    parts = re.split(r"[._-]", local)
    # Only attempt if we have ≥2 parts that look like name components
    long_parts = [p.capitalize() for p in parts if len(p) >= 2]
    if len(long_parts) >= 2:
        return " ".join(long_parts)
    return ""
