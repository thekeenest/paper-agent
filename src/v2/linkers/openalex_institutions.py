"""
OpenAlex institution linker.

Provides three entry points used by the Critic:

* ``by_ror_id(ror_id)``   — resolve a ROR ID to an OpenAlex Institution record
* ``search(text)``         — free-text institution search
* ``by_paper(id_or_doi)`` — fetch author → affiliation rows for a single work

OpenAlex is free and open; no key required, but requests should include a
polite-pool email (``OPENALEX_EMAIL`` env var) for higher rate limits.

Usage
-----
::

    from src.v2.linkers.openalex_institutions import OpenAlexInstitutions

    oa = OpenAlexInstitutions()
    inst = await oa.by_ror_id("https://ror.org/042nb2s44")
    print(inst.display_name, inst.oa_id)

References
----------
  OpenAlex API: https://docs.openalex.org
"""
from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import quote

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

_OA_BASE = "https://api.openalex.org"
_DEFAULT_TTL = 3 * 86_400  # 3 days


# ─────────────────────────── data models ────────────────────────────────────


class Institution(BaseModel):
    model_config = ConfigDict(frozen=True)

    oa_id: str = Field(description="OpenAlex institution ID, e.g. I27837315")
    display_name: str
    ror_id: str | None = None
    country_code: str | None = None
    type: str | None = None
    provenance: Literal["openalex"] = "openalex"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Authorship(BaseModel):
    model_config = ConfigDict(frozen=True)

    author_name: str
    author_oa_id: str | None = None
    institutions: list[Institution]
    provenance: Literal["openalex"] = "openalex"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─────────────────────────── linker ─────────────────────────────────────────


class OpenAlexInstitutions:
    """Async OpenAlex institution resolver.

    Parameters
    ----------
    email:
        Passed as ``mailto=`` for the polite pool (higher rate limits).
        Falls back to ``OPENALEX_EMAIL`` env var, then omitted.
    timeout:
        HTTP timeout in seconds.
    cache_ttl:
        Cache entry lifetime in seconds.  Default 3 days.
    """

    def __init__(
        self,
        email: str | None = None,
        timeout: float = 10.0,
        cache_ttl: float = _DEFAULT_TTL,
    ) -> None:
        self._email = email or os.getenv("OPENALEX_EMAIL")
        self._timeout = timeout
        self._cache = Cache(namespace="openalex", ttl_seconds=cache_ttl)

    # ── public API ────────────────────────────────────────────────────────

    async def by_ror_id(self, ror_id: str) -> Institution | None:
        """Return the OpenAlex Institution whose ``ror`` field matches *ror_id*.

        Returns ``None`` if not found (no exception).
        """
        cache_key = f"ror:{ror_id}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return Institution.model_validate(cached) if cached else None

        data = await self._get(f"/institutions/ror:{quote(ror_id, safe='')}")
        if data is None:
            await self._cache.set(cache_key, {})
            return None
        inst = _parse_institution(data)
        await self._cache.set(cache_key, inst.model_dump(mode="json"))
        return inst

    async def search(self, text: str) -> list[Institution]:
        """Free-text institution search (OpenAlex ``/institutions?search=``)."""
        cache_key = f"search:{text}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return [Institution.model_validate(i) for i in cached]

        data = await self._get("/institutions", params={"search": text, "per-page": "10"})
        if data is None:
            return []

        results = data.get("results", [])
        institutions = []
        for item in results:
            try:
                institutions.append(_parse_institution(item))
            except (KeyError, TypeError) as exc:
                _LOG.warning("OpenAlex: skipping malformed institution: %s", exc)

        await self._cache.set(cache_key, [i.model_dump(mode="json") for i in institutions])
        return institutions

    async def by_paper(self, openalex_id_or_doi: str) -> list[Authorship]:
        """Fetch author→affiliation rows for a single work.

        Accepts either an OpenAlex work ID (``W1234567``) or a DOI
        (``https://doi.org/10.xxx`` or bare ``10.xxx``).
        """
        cache_key = f"paper:{openalex_id_or_doi}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return [Authorship.model_validate(a) for a in cached]

        work_id = _normalise_work_id(openalex_id_or_doi)
        data = await self._get(f"/works/{work_id}")
        if data is None:
            return []

        authorships = _parse_authorships(data)
        await self._cache.set(
            cache_key, [a.model_dump(mode="json") for a in authorships]
        )
        return authorships

    # ── internals ────────────────────────────────────────────────────────

    def _build_params(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        params: dict[str, str] = {}
        if self._email:
            params["mailto"] = self._email
        if extra:
            params.update(extra)
        return params

    @retry(
        retry=retry_if_exception_type(NetworkError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any] | None:
        url = f"{_OA_BASE}{path}"
        merged = self._build_params(params)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(url, params=merged)
            except httpx.TransportError as exc:
                raise NetworkError("openalex", str(exc)) from exc

        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 30))
            raise RateLimitError("openalex", "429", retry_after=retry_after)
        if resp.status_code >= 500:
            raise NetworkError("openalex", f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise NetworkError("openalex", f"HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data: dict[str, Any] = resp.json()
            return data
        except Exception as exc:
            raise ParseError("openalex", f"Invalid JSON: {exc}") from exc


# ─────────────────────────── parsers ────────────────────────────────────────


def _parse_institution(data: dict[str, Any]) -> Institution:
    oa_id = data.get("id", "")
    # Strip full URL prefix if present
    if oa_id.startswith("https://openalex.org/"):
        oa_id = oa_id[len("https://openalex.org/"):]

    ror_raw = data.get("ror")
    ror_id: str | None = None
    if ror_raw:
        ror_id = ror_raw if ror_raw.startswith("https://") else f"https://ror.org/{ror_raw}"

    return Institution(
        oa_id=oa_id,
        display_name=data.get("display_name", ""),
        ror_id=ror_id,
        country_code=data.get("country_code"),
        type=data.get("type"),
    )


def _parse_authorships(data: dict[str, Any]) -> list[Authorship]:
    rows = data.get("authorships", [])
    result: list[Authorship] = []
    for row in rows:
        author = row.get("author", {})
        name = author.get("display_name", "")
        raw_id = author.get("id", "")
        if raw_id.startswith("https://openalex.org/"):
            raw_id = raw_id[len("https://openalex.org/"):]

        insts: list[Institution] = []
        for inst_data in row.get("institutions", []):
            with contextlib.suppress(KeyError, TypeError):
                insts.append(_parse_institution(inst_data))

        result.append(
            Authorship(
                author_name=name,
                author_oa_id=raw_id or None,
                institutions=insts,
            )
        )
    return result


def _normalise_work_id(raw: str) -> str:
    """Convert DOI or full URL to a form the OA API accepts."""
    if raw.startswith("https://openalex.org/"):
        return raw[len("https://openalex.org/"):]
    if raw.startswith("https://doi.org/"):
        return f"doi:{raw[len('https://doi.org/')]}"
    if raw.startswith("10."):
        return f"doi:{raw}"
    return raw
