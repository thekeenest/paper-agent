"""
Source router — resolves a query / DOI / ArXiv ID to a CanonicalPaper
and downloads the PDF.

Supported sources (in priority order):
  1. ArXiv      — native arxiv library; direct PDF URL
  2. OpenAlex   — works/{id}; PDF via open-access URL
  3. Semantic Scholar — /graph/v1/paper/{id}; openAccessPdf field
  4. ACL Anthology — scrapes the ACL Anthology HTML (best-effort)
  5. OpenReview — optional; set OPENREVIEW_USERNAME + OPENREVIEW_PASSWORD

Usage
-----
::

    from src.v2.orchestration.source_router import SourceRouter

    router = SourceRouter()
    paper, pdf_path = await router.resolve("2301.00001")
    # or
    paper, pdf_path = await router.resolve("cat:cs.AI", n=1)
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx
import structlog

from .contracts import CanonicalPaper

_LOG = structlog.get_logger(__name__)

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_OA_BASE = "https://api.openalex.org"
_ACL_BASE = "https://aclanthology.org"

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


# ─────────────────────────── public API ─────────────────────────────────────


class SourceRouter:
    """Multi-source paper resolver.

    Parameters
    ----------
    download_dir:
        Where to save downloaded PDFs.  Defaults to a temp directory.
    timeout:
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        download_dir: str | Path | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._dir = Path(download_dir) if download_dir else Path(tempfile.mkdtemp(prefix="paper_agent_"))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout

    async def resolve(
        self,
        query: str,
        n: int = 1,
    ) -> list[tuple[CanonicalPaper, Path]]:
        """Resolve *query* to *(paper, pdf_path)* pairs.

        Tries sources in priority order until one succeeds.

        Parameters
        ----------
        query:
            ArXiv ID (``2301.00001``), ArXiv category query (``cat:cs.AI``),
            DOI, OpenAlex work ID (``W...``), or free-text title.
        n:
            Maximum number of papers to return.
        """
        _LOG.info("source_router.resolve", query=query, n=n)

        # ArXiv ID
        bare = query.strip()
        if _ARXIV_ID_RE.match(bare) or bare.startswith("arxiv:"):
            arxiv_id = bare.removeprefix("arxiv:")
            results = await self._from_arxiv_id(arxiv_id)
            if results:
                return results[:n]

        # ArXiv category / keyword search
        results = await self._from_arxiv_search(query, n=n)
        if results:
            return results[:n]

        # OpenAlex work ID
        if bare.startswith("W") and bare[1:].isdigit():
            results = await self._from_openalex_id(bare)
            if results:
                return results[:n]

        # DOI
        if bare.startswith("10.") or bare.startswith("https://doi.org/"):
            doi = bare.removeprefix("https://doi.org/")
            results = await self._from_s2_doi(doi)
            if results:
                return results[:n]

        # Fallback: OpenAlex title search
        results = await self._from_openalex_search(query, n=n)
        if results:
            return results[:n]

        _LOG.warning("source_router.no_results", query=query)
        return []

    # ── ArXiv ────────────────────────────────────────────────────────────

    async def _from_arxiv_id(self, arxiv_id: str) -> list[tuple[CanonicalPaper, Path]]:
        try:
            import arxiv
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(client.results(search))
            if not results:
                return []
            r = results[0]
            paper = CanonicalPaper(
                paper_id=f"arxiv:{r.get_short_id()}",
                title=r.title,
                abstract=r.summary,
                venue=None,
                year=r.published.year if r.published else None,
                pdf_url=r.pdf_url,
                source="arxiv",
                raw_metadata={"authors": [str(a) for a in r.authors]},
            )
            pdf_path = await self._download_pdf(r.pdf_url, f"arxiv_{r.get_short_id()}.pdf")
            if pdf_path:
                return [(paper, pdf_path)]
        except Exception as exc:
            _LOG.warning("source_router.arxiv_id_failed", arxiv_id=arxiv_id, error=str(exc))
        return []

    async def _from_arxiv_search(self, query: str, n: int) -> list[tuple[CanonicalPaper, Path]]:
        try:
            import arxiv
            client = arxiv.Client()
            search = arxiv.Search(query=query, max_results=n)
            out: list[tuple[CanonicalPaper, Path]] = []
            for r in client.results(search):
                paper = CanonicalPaper(
                    paper_id=f"arxiv:{r.get_short_id()}",
                    title=r.title,
                    abstract=r.summary,
                    year=r.published.year if r.published else None,
                    pdf_url=r.pdf_url,
                    source="arxiv",
                    raw_metadata={"authors": [str(a) for a in r.authors]},
                )
                pdf_path = await self._download_pdf(r.pdf_url, f"arxiv_{r.get_short_id()}.pdf")
                if pdf_path:
                    out.append((paper, pdf_path))
                    if len(out) >= n:
                        break
            return out
        except Exception as exc:
            _LOG.warning("source_router.arxiv_search_failed", query=query, error=str(exc))
        return []

    # ── OpenAlex ─────────────────────────────────────────────────────────

    async def _from_openalex_id(self, work_id: str) -> list[tuple[CanonicalPaper, Path]]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{_OA_BASE}/works/{work_id}")
            if resp.status_code != 200:
                return []
            data = resp.json()
            paper, pdf_url = _openalex_to_canonical(data)
            if pdf_url:
                pdf_path = await self._download_pdf(pdf_url, f"oa_{work_id}.pdf")
                if pdf_path:
                    return [(paper, pdf_path)]
        except Exception as exc:
            _LOG.warning("source_router.openalex_id_failed", work_id=work_id, error=str(exc))
        return []

    async def _from_openalex_search(self, query: str, n: int) -> list[tuple[CanonicalPaper, Path]]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{_OA_BASE}/works",
                    params={"search": query, "per-page": str(n), "filter": "has_oa_accepted_or_published_version:true"},
                )
            if resp.status_code != 200:
                return []
            out: list[tuple[CanonicalPaper, Path]] = []
            for item in resp.json().get("results", []):
                paper, pdf_url = _openalex_to_canonical(item)
                if pdf_url:
                    pdf_path = await self._download_pdf(pdf_url, f"oa_{paper.paper_id.split(':')[1]}.pdf")
                    if pdf_path:
                        out.append((paper, pdf_path))
                        if len(out) >= n:
                            break
            return out
        except Exception as exc:
            _LOG.warning("source_router.openalex_search_failed", query=query, error=str(exc))
        return []

    # ── Semantic Scholar ──────────────────────────────────────────────────

    async def _from_s2_doi(self, doi: str) -> list[tuple[CanonicalPaper, Path]]:
        try:
            api_key = os.getenv("S2_API_KEY")
            headers = {"x-api-key": api_key} if api_key else {}
            async with httpx.AsyncClient(timeout=self._timeout, headers=headers) as client:
                resp = await client.get(
                    f"{_S2_BASE}/paper/DOI:{doi}",
                    params={"fields": "title,abstract,year,venue,openAccessPdf,externalIds"},
                )
            if resp.status_code != 200:
                return []
            data = resp.json()
            paper = CanonicalPaper(
                paper_id=f"doi:{doi}",
                title=data.get("title", ""),
                abstract=data.get("abstract", ""),
                venue=data.get("venue"),
                year=data.get("year"),
                pdf_url=(data.get("openAccessPdf") or {}).get("url"),
                source="s2",
                raw_metadata=data,
            )
            if paper.pdf_url:
                pdf_path = await self._download_pdf(paper.pdf_url, f"s2_{doi.replace('/', '_')}.pdf")
                if pdf_path:
                    return [(paper, pdf_path)]
        except Exception as exc:
            _LOG.warning("source_router.s2_doi_failed", doi=doi, error=str(exc))
        return []

    # ── ACL Anthology ─────────────────────────────────────────────────────

    async def _from_acl(self, acl_id: str) -> list[tuple[CanonicalPaper, Path]]:
        """Fetch paper metadata and PDF from ACL Anthology."""
        try:
            pdf_url = f"{_ACL_BASE}/{acl_id}.pdf"
            # Metadata via the ACL Anthology JSON endpoint
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{_ACL_BASE}/{acl_id}.json")
            if resp.status_code != 200:
                return []
            data = resp.json()
            paper = CanonicalPaper(
                paper_id=f"acl:{acl_id}",
                title=data.get("title", ""),
                abstract=data.get("abstract", ""),
                venue=data.get("booktitle"),
                year=int(data["year"]) if data.get("year") else None,
                pdf_url=pdf_url,
                source="acl",
                raw_metadata=data,
            )
            pdf_path = await self._download_pdf(pdf_url, f"acl_{acl_id.replace('/', '_')}.pdf")
            if pdf_path:
                return [(paper, pdf_path)]
        except Exception as exc:
            _LOG.warning("source_router.acl_failed", acl_id=acl_id, error=str(exc))
        return []

    # ── OpenReview (optional) ─────────────────────────────────────────────

    async def _from_openreview(self, note_id: str) -> list[tuple[CanonicalPaper, Path]]:
        """Resolve an OpenReview note ID to a paper + PDF."""
        username = os.getenv("OPENREVIEW_USERNAME")
        password = os.getenv("OPENREVIEW_PASSWORD")
        try:
            import openreview
            client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net",
                username=username,
                password=password,
            )
            note = client.get_note(note_id)
            content = note.content or {}
            title = _or_str(content.get("title"))
            abstract = _or_str(content.get("abstract"))
            pdf_url = f"https://openreview.net/pdf?id={note_id}"
            paper = CanonicalPaper(
                paper_id=f"openreview:{note_id}",
                title=title,
                abstract=abstract,
                source="openreview",
                raw_metadata={"note_id": note_id},
            )
            pdf_path = await self._download_pdf(pdf_url, f"or_{note_id}.pdf")
            if pdf_path:
                return [(paper, pdf_path)]
        except ImportError:
            _LOG.debug("source_router.openreview_not_installed")
        except Exception as exc:
            _LOG.warning("source_router.openreview_failed", note_id=note_id, error=str(exc))
        return []

    # ── PDF download ──────────────────────────────────────────────────────

    async def _download_pdf(self, url: str, filename: str) -> Path | None:
        dest = self._dir / filename
        if dest.exists():
            _LOG.debug("source_router.pdf_cached", path=str(dest))
            return dest
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": "paper-agent/2.0 (research; mailto:research@example.com)"},
            ) as client:
                resp = await client.get(url)
            if resp.status_code == 200 and b"%PDF" in resp.content[:8]:
                dest.write_bytes(resp.content)
                _LOG.info("source_router.pdf_downloaded", path=str(dest), bytes=len(resp.content))
                return dest
            _LOG.warning("source_router.pdf_not_pdf", url=url, status=resp.status_code)
        except Exception as exc:
            _LOG.warning("source_router.pdf_download_failed", url=url, error=str(exc))
        return None


# ─────────────────────────── helpers ────────────────────────────────────────


def _openalex_to_canonical(data: dict[str, Any]) -> tuple[CanonicalPaper, str | None]:
    oa_id = (data.get("id") or "").removeprefix("https://openalex.org/")
    title = data.get("display_name") or data.get("title") or ""
    abstract = _reconstruct_abstract(data.get("abstract_inverted_index"))
    year = data.get("publication_year")
    venue_data = data.get("primary_location") or {}
    source_data = venue_data.get("source") or {}
    venue = source_data.get("display_name")
    pdf_url = (venue_data.get("pdf_url")) or None
    if not pdf_url:
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("pdf_url")

    paper = CanonicalPaper(
        paper_id=f"openalex:{oa_id}",
        title=title,
        abstract=abstract,
        venue=venue,
        year=year,
        pdf_url=pdf_url,
        source="openalex",
        raw_metadata=data,
    )
    return paper, pdf_url


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inverted.items():
        for p in pos_list:
            positions[p] = word
    return " ".join(positions[i] for i in sorted(positions))


def _or_str(v: Any) -> str:
    if isinstance(v, dict):
        value: str = v.get("value", "")
        return value
    return str(v) if v else ""
