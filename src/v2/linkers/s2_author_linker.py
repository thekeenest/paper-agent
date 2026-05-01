"""
Semantic Scholar (S2) author linker.

Matches an author name (+ optional paper context) to a Semantic Scholar
author record via the public ``/graph/v1/author/search`` endpoint.

Rate limits
-----------
* Without API key : ~100 requests / 5 min  (~0.33 RPS)
* With API key    : ~1 RPS

Set ``S2_API_KEY`` env var to use the authenticated pool.

Usage
-----
::

    from src.v2.linkers.s2_author_linker import S2AuthorLinker

    linker = S2AuthorLinker()
    authors = await linker.match("Alice Smith", paper_s2_id="abc123")
    for a in authors:
        print(a.s2_id, a.name, a.h_index)

References
----------
  S2 API docs: https://api.semanticscholar.org/graph/v1
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._cache import Cache
from ._errors import NetworkError, ParseError, RateLimitError

_LOG = logging.getLogger(__name__)

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_DEFAULT_TTL = 2 * 86_400  # 2 days

# Adaptive rate limiting: track last request time per instance
_AUTHOR_FIELDS = "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex"


# ─────────────────────────── data models ────────────────────────────────────


class S2Author(BaseModel):
    model_config = ConfigDict(frozen=True)

    s2_id: str = Field(description="Semantic Scholar author ID")
    name: str
    affiliations: list[str] = Field(default_factory=list)
    homepage: str | None = None
    paper_count: int | None = None
    citation_count: int | None = None
    h_index: int | None = None
    provenance: Literal["s2"] = "s2"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─────────────────────────── rate limiter ───────────────────────────────────


class _RateLimiter:
    """Token-bucket rate limiter (synchronous, used in async context)."""

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps
        self._last: float = 0.0

    async def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        wait_s = self._min_interval - elapsed
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        self._last = time.monotonic()


# ─────────────────────────── linker ─────────────────────────────────────────


class S2AuthorLinker:
    """Async Semantic Scholar author resolver.

    Parameters
    ----------
    api_key:
        S2 API key.  Falls back to ``S2_API_KEY`` env var, then unauthenticated.
    timeout:
        HTTP timeout in seconds.
    cache_ttl:
        Cache entry lifetime in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
        cache_ttl: float = _DEFAULT_TTL,
    ) -> None:
        self._api_key = api_key or os.getenv("S2_API_KEY")
        self._timeout = timeout
        self._cache = Cache(namespace="s2_author", ttl_seconds=cache_ttl)
        # 1 RPS with key, 0.33 RPS without
        rps = 1.0 if self._api_key else 100.0 / 300.0
        self._rate = _RateLimiter(rps)

    async def match(
        self,
        name: str,
        paper_s2_id: str | None = None,
    ) -> list[S2Author]:
        """Search for S2 authors matching *name*.

        Parameters
        ----------
        name:
            Author display name (e.g. ``"Alice Smith"``).
        paper_s2_id:
            If provided, results are filtered to authors who have a paper
            with this S2 paper ID (best-effort, applied client-side).

        Returns
        -------
        list[S2Author]
            Up to 10 candidates, ordered by S2's relevance ranking.
        """
        cache_key = f"{name}|{paper_s2_id or ''}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            _LOG.debug("S2 cache hit for %r", name)
            return [S2Author.model_validate(a) for a in cached]

        raw = await self._fetch_search(name)
        authors = _parse_author_search(raw)

        if paper_s2_id:
            authors = await self._filter_by_paper(authors, paper_s2_id)

        await self._cache.set(cache_key, [a.model_dump(mode="json") for a in authors])
        return authors

    # ── internals ────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._api_key:
            h["x-api-key"] = self._api_key
        return h

    @retry(
        retry=retry_if_exception_type(NetworkError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    async def _fetch_search(self, name: str) -> dict[str, Any]:
        await self._rate.wait()
        url = f"{_S2_BASE}/author/search"
        params = {"query": name, "fields": _AUTHOR_FIELDS, "limit": "10"}
        async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers()) as client:
            try:
                resp = await client.get(url, params=params)
            except httpx.TransportError as exc:
                raise NetworkError("s2", str(exc)) from exc

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 60))
            raise RateLimitError("s2", "429 Too Many Requests", retry_after=retry_after)
        if resp.status_code >= 500:
            raise NetworkError("s2", f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise NetworkError("s2", f"HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data: dict[str, Any] = resp.json()
            return data
        except Exception as exc:
            raise ParseError("s2", f"Invalid JSON: {exc}") from exc

    async def _filter_by_paper(
        self, authors: list[S2Author], paper_id: str
    ) -> list[S2Author]:
        """Keep only authors who have co-authored *paper_id* (best-effort)."""
        await self._rate.wait()
        url = f"{_S2_BASE}/paper/{paper_id}"
        params = {"fields": "authors"}
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, headers=self._headers()
            ) as client:
                resp = await client.get(url, params=params)
            if resp.status_code == 200:
                paper_author_ids = {
                    a.get("authorId")
                    for a in resp.json().get("authors", [])
                    if a.get("authorId")
                }
                if paper_author_ids:
                    filtered = [a for a in authors if a.s2_id in paper_author_ids]
                    return filtered if filtered else authors
        except Exception as exc:
            _LOG.debug("S2 paper filter failed for %s: %s", paper_id, exc)
        return authors


# ─────────────────────────── parsers ────────────────────────────────────────


def _parse_author_search(data: dict[str, Any]) -> list[S2Author]:
    items = data.get("data", [])
    if not isinstance(items, list):
        raise ParseError("s2", f"Unexpected shape: {list(data.keys())}")

    authors: list[S2Author] = []
    for item in items:
        try:
            s2_id = item.get("authorId", "")
            if not s2_id:
                continue
            authors.append(
                S2Author(
                    s2_id=s2_id,
                    name=item.get("name", ""),
                    affiliations=item.get("affiliations") or [],
                    homepage=item.get("homepage") or None,
                    paper_count=item.get("paperCount"),
                    citation_count=item.get("citationCount"),
                    h_index=item.get("hIndex"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            _LOG.warning("S2: skipping malformed author: %s", exc)

    return authors
