"""
ROR (Research Organization Registry) linker.

Resolves free-text affiliation strings to ROR IDs via the public
``/organizations?affiliation=`` endpoint.

ROR has **no rate limit** and requires no API key.

Usage
-----
::

    from src.v2.linkers.ror_linker import RORLinker

    linker = RORLinker()
    matches = await linker.find("MIT CSAIL, Cambridge MA")
    for m in matches:
        print(m.ror_id, m.name, m.score)

CLI smoke
---------
::

    python -m src.v2.linkers.ror_linker "Google DeepMind, London"

References
----------
  ROR API docs: https://ror.readme.io/docs/affiliation-matching
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
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
from ._errors import NetworkError, NotFoundError, ParseError, RateLimitError

_LOG = logging.getLogger(__name__)

_ROR_BASE = "https://api.ror.org/v2"
_DEFAULT_TTL = 7 * 86_400  # 7 days — ROR data is stable


# ─────────────────────────── data models ────────────────────────────────────


class ROR_Match(BaseModel):
    model_config = ConfigDict(frozen=True)

    ror_id: str = Field(description="Full ROR URL, e.g. https://ror.org/042nb2s44")
    name: str
    country_code: str | None = None
    country_name: str | None = None
    score: float = Field(ge=0.0, le=1.0, description="Affiliation match confidence")
    provenance: Literal["ror"] = "ror"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─────────────────────────── linker ─────────────────────────────────────────


class RORLinker:
    """Async-first ROR affiliation resolver.

    Parameters
    ----------
    timeout:
        HTTP timeout in seconds.
    cache_ttl:
        How long (seconds) to cache responses.  Default 7 days.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        cache_ttl: float = _DEFAULT_TTL,
    ) -> None:
        self._timeout = timeout
        self._cache = Cache(namespace="ror", ttl_seconds=cache_ttl)

    async def find(
        self,
        name: str,
        country_hint: str | None = None,
    ) -> list[ROR_Match]:
        """Match *name* against ROR's affiliation endpoint.

        Parameters
        ----------
        name:
            Raw affiliation string (e.g. ``"MIT CSAIL, Cambridge MA 02139"``).
        country_hint:
            ISO-3166 two-letter country code to bias results (optional).

        Returns
        -------
        list[ROR_Match]
            Sorted by score descending.  Empty list if no match found.

        Raises
        ------
        RateLimitError
            If ROR returns 429.
        NetworkError
            On connection / timeout failure.
        ParseError
            If the API returns unexpected JSON.
        """
        cache_key = f"{name}|{country_hint or ''}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            _LOG.debug("ROR cache hit for %r", name)
            return [ROR_Match.model_validate(m) for m in cached]

        raw = await self._fetch_affiliation(name)
        matches = _parse_affiliation_response(raw)

        if country_hint:
            preferred = [m for m in matches if m.country_code == country_hint.upper()]
            rest = [m for m in matches if m.country_code != country_hint.upper()]
            matches = preferred + rest

        await self._cache.set(cache_key, [m.model_dump(mode="json") for m in matches])
        return matches

    @retry(
        retry=retry_if_exception_type(NetworkError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _fetch_affiliation(self, affiliation: str) -> dict[str, Any]:
        url = f"{_ROR_BASE}/organizations"
        params = {"affiliation": affiliation}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(url, params=params)
            except httpx.TransportError as exc:
                raise NetworkError("ror", str(exc)) from exc

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 60))
            raise RateLimitError("ror", "429 Too Many Requests", retry_after=retry_after)
        if resp.status_code >= 500:
            raise NetworkError("ror", f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise NetworkError("ror", f"HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data: dict[str, Any] = resp.json()
            return data
        except Exception as exc:
            raise ParseError("ror", f"Invalid JSON: {exc}") from exc


# ─────────────────────────── response parsing ───────────────────────────────


def _parse_affiliation_response(data: dict[str, Any]) -> list[ROR_Match]:
    """Parse the ROR v2 /organizations?affiliation= response."""
    items = data.get("items")
    if not isinstance(items, list):
        raise ParseError("ror", f"Unexpected response shape: {list(data.keys())}")

    matches: list[ROR_Match] = []
    for item in items:
        try:
            org = item.get("organization", {})
            ror_id: str = org.get("id", "")
            if not ror_id:
                continue

            # Primary name
            names = org.get("names", [])
            name = next(
                (n["value"] for n in names if "ror_display" in n.get("types", [])),
                names[0]["value"] if names else ror_id,
            )

            # Country
            locations = org.get("locations", [])
            country_code: str | None = None
            country_name: str | None = None
            if locations:
                geonames = locations[0].get("geonames_details", {})
                country_code = geonames.get("country_code")
                country_name = geonames.get("country_name")

            score = float(item.get("score", 0.0))
            matches.append(
                ROR_Match(
                    ror_id=ror_id,
                    name=name,
                    country_code=country_code,
                    country_name=country_name,
                    score=score,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            _LOG.warning("ROR: skipping malformed item: %s", exc)

    return sorted(matches, key=lambda m: m.score, reverse=True)


# ─────────────────────────── CLI ─────────────────────────────────────────────


async def _cli_main(query: str) -> None:
    linker = RORLinker()
    matches = await linker.find(query)
    if not matches:
        print(f"No ROR matches for: {query!r}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps([m.model_dump(mode="json") for m in matches], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.v2.linkers.ror_linker <affiliation string>", file=sys.stderr)
        sys.exit(1)
    asyncio.run(_cli_main(" ".join(sys.argv[1:])))
