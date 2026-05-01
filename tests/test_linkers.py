"""
Tests for src.v2.linkers — 12 tests, 3 per linker.

All network calls are intercepted by pytest-httpx so no real HTTP is made.
The cache database is redirected to a tmp path per test session.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx
import pytest
from pytest_httpx import HTTPXMock

# ── redirect cache to a tmp file so tests never share state ──────────────────


@pytest.fixture(autouse=True, scope="session")
def _tmp_cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("linker_cache") / "test_cache.sqlite"
    os.environ["LINKER_CACHE_PATH"] = str(tmp)
    yield
    os.environ.pop("LINKER_CACHE_PATH", None)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: recorded API responses
# ─────────────────────────────────────────────────────────────────────────────

# ── ROR /v2/organizations?affiliation= ───────────────────────────────────────

ROR_FIXTURE_MIT = {
    "items": [
        {
            "score": 0.97,
            "organization": {
                "id": "https://ror.org/042nb2s44",
                "names": [{"value": "Massachusetts Institute of Technology", "types": ["ror_display"]}],
                "locations": [
                    {
                        "geonames_details": {
                            "country_code": "US",
                            "country_name": "United States",
                        }
                    }
                ],
            },
        },
        {
            "score": 0.61,
            "organization": {
                "id": "https://ror.org/02jx3x895",
                "names": [{"value": "MIT Lincoln Laboratory", "types": ["ror_display"]}],
                "locations": [
                    {
                        "geonames_details": {
                            "country_code": "US",
                            "country_name": "United States",
                        }
                    }
                ],
            },
        },
    ]
}

ROR_FIXTURE_DEEPMIND = {
    "items": [
        {
            "score": 0.95,
            "organization": {
                "id": "https://ror.org/05k73zm34",
                "names": [{"value": "Google DeepMind", "types": ["ror_display"]}],
                "locations": [
                    {
                        "geonames_details": {
                            "country_code": "GB",
                            "country_name": "United Kingdom",
                        }
                    }
                ],
            },
        }
    ]
}

ROR_FIXTURE_EMPTY = {"items": []}

# ── OpenAlex institutions ─────────────────────────────────────────────────────

OA_INSTITUTION_MIT = {
    "id": "https://openalex.org/I63966007",
    "display_name": "Massachusetts Institute of Technology",
    "ror": "https://ror.org/042nb2s44",
    "country_code": "US",
    "type": "education",
}

OA_SEARCH_RESULT = {
    "results": [OA_INSTITUTION_MIT],
    "meta": {"count": 1},
}

OA_WORK_AUTHORSHIPS = {
    "authorships": [
        {
            "author": {
                "id": "https://openalex.org/A1234567",
                "display_name": "Alice Smith",
            },
            "institutions": [OA_INSTITUTION_MIT],
        },
        {
            "author": {
                "id": "https://openalex.org/A7654321",
                "display_name": "Bob Jones",
            },
            "institutions": [],
        },
    ]
}

# ── Semantic Scholar author search ───────────────────────────────────────────

S2_AUTHOR_FIXTURE = {
    "data": [
        {
            "authorId": "144783904",
            "name": "Alice Smith",
            "affiliations": ["MIT CSAIL"],
            "homepage": None,
            "paperCount": 42,
            "citationCount": 1200,
            "hIndex": 18,
        },
        {
            "authorId": "9876543",
            "name": "Alice Smith",
            "affiliations": ["Stanford University"],
            "homepage": None,
            "paperCount": 5,
            "citationCount": 30,
            "hIndex": 3,
        },
    ]
}

S2_AUTHOR_EMPTY = {"data": []}

S2_AUTHOR_WITH_PAPER = {
    "data": [
        {
            "authorId": "144783904",
            "name": "Alice Smith",
            "affiliations": ["MIT CSAIL"],
            "homepage": None,
            "paperCount": 42,
            "citationCount": 1200,
            "hIndex": 18,
        }
    ]
}


# ─────────────────────────────────────────────────────────────────────────────
# ROR linker tests (3 tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestRORLinker:

    def test_find_returns_sorted_matches(self, httpx_mock: HTTPXMock):
        """find() returns ROR_Match objects sorted by score desc."""
        from src.v2.linkers.ror_linker import RORLinker

        httpx_mock.add_response(
            url="https://api.ror.org/v2/organizations?affiliation=MIT+CSAIL",
            json=ROR_FIXTURE_MIT,
        )

        linker = RORLinker(cache_ttl=0)  # TTL=0 → always miss cache
        matches = asyncio.run(linker.find("MIT CSAIL"))

        assert len(matches) == 2
        assert matches[0].ror_id == "https://ror.org/042nb2s44"
        assert matches[0].score == pytest.approx(0.97)
        assert matches[0].provenance == "ror"
        assert matches[0].country_code == "US"
        assert matches[0].retrieved_at is not None

    def test_find_empty_returns_empty_list(self, httpx_mock: HTTPXMock):
        """find() returns [] when ROR returns no items (no exception)."""
        from src.v2.linkers.ror_linker import RORLinker

        httpx_mock.add_response(
            url="https://api.ror.org/v2/organizations?affiliation=xyzzy+nonexistent",
            json=ROR_FIXTURE_EMPTY,
        )

        linker = RORLinker(cache_ttl=0)
        matches = asyncio.run(linker.find("xyzzy nonexistent"))
        assert matches == []

    def test_find_raises_network_error_on_500(self, httpx_mock: HTTPXMock):
        """NetworkError is raised (not swallowed) on 5xx responses."""
        from src.v2.linkers._errors import NetworkError
        from src.v2.linkers.ror_linker import RORLinker

        # Return 500 for all 3 retry attempts
        for _ in range(3):
            httpx_mock.add_response(
                url="https://api.ror.org/v2/organizations?affiliation=Google+DeepMind",
                status_code=500,
            )

        linker = RORLinker(cache_ttl=0)
        with pytest.raises(NetworkError):
            asyncio.run(linker.find("Google DeepMind"))


# ─────────────────────────────────────────────────────────────────────────────
# OpenAlex linker tests (3 tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenAlexInstitutions:

    def test_by_ror_id_returns_institution(self, httpx_mock: HTTPXMock):
        """by_ror_id() resolves a known ROR ID to an Institution."""
        from src.v2.linkers.openalex_institutions import OpenAlexInstitutions

        httpx_mock.add_response(
            url="https://api.openalex.org/institutions/ror:https%3A%2F%2Fror.org%2F042nb2s44",
            json=OA_INSTITUTION_MIT,
        )

        oa = OpenAlexInstitutions(cache_ttl=0)
        inst = asyncio.run(oa.by_ror_id("https://ror.org/042nb2s44"))

        assert inst is not None
        assert inst.oa_id == "I63966007"
        assert inst.display_name == "Massachusetts Institute of Technology"
        assert inst.ror_id == "https://ror.org/042nb2s44"
        assert inst.provenance == "openalex"

    def test_search_returns_list_of_institutions(self, httpx_mock: HTTPXMock):
        """search() wraps /institutions?search= and returns Institution list."""
        from src.v2.linkers.openalex_institutions import OpenAlexInstitutions

        httpx_mock.add_response(
            url=re.compile(r"https://api\.openalex\.org/institutions\?.*"),
            json=OA_SEARCH_RESULT,
        )

        oa = OpenAlexInstitutions(cache_ttl=0)
        results = asyncio.run(oa.search("MIT"))

        assert len(results) == 1
        assert results[0].display_name == "Massachusetts Institute of Technology"
        assert results[0].country_code == "US"

    def test_by_paper_returns_authorships(self, httpx_mock: HTTPXMock):
        """by_paper() returns Authorship rows with nested institutions."""
        from src.v2.linkers.openalex_institutions import OpenAlexInstitutions

        httpx_mock.add_response(
            url="https://api.openalex.org/works/W1234567",
            json=OA_WORK_AUTHORSHIPS,
        )

        oa = OpenAlexInstitutions(cache_ttl=0)
        authorships = asyncio.run(oa.by_paper("W1234567"))

        assert len(authorships) == 2
        alice = next(a for a in authorships if a.author_name == "Alice Smith")
        assert alice.author_oa_id == "A1234567"
        assert len(alice.institutions) == 1
        assert alice.institutions[0].display_name == "Massachusetts Institute of Technology"
        assert alice.provenance == "openalex"


# ─────────────────────────────────────────────────────────────────────────────
# S2 author linker tests (3 tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestS2AuthorLinker:

    def test_match_returns_s2author_list(self, httpx_mock: HTTPXMock):
        """match() returns S2Author objects with correct fields."""
        from src.v2.linkers.s2_author_linker import S2AuthorLinker

        httpx_mock.add_response(
            url=re.compile(r"https://api\.semanticscholar\.org/graph/v1/author/search.*"),
            json=S2_AUTHOR_FIXTURE,
        )

        linker = S2AuthorLinker(cache_ttl=0)
        authors = asyncio.run(linker.match("Alice Smith"))

        assert len(authors) == 2
        top = authors[0]
        assert top.s2_id == "144783904"
        assert top.name == "Alice Smith"
        assert "MIT CSAIL" in top.affiliations
        assert top.h_index == 18
        assert top.provenance == "s2"
        assert top.retrieved_at is not None

    def test_match_empty_result(self, httpx_mock: HTTPXMock):
        """match() returns [] for an unknown author (no exception)."""
        from src.v2.linkers.s2_author_linker import S2AuthorLinker

        httpx_mock.add_response(
            url=re.compile(r"https://api\.semanticscholar\.org/graph/v1/author/search.*"),
            json=S2_AUTHOR_EMPTY,
        )

        linker = S2AuthorLinker(cache_ttl=0)
        authors = asyncio.run(linker.match("Zxqwerty Nonexistent"))
        assert authors == []

    def test_match_raises_rate_limit_error_on_429(self, httpx_mock: HTTPXMock):
        """RateLimitError is raised on 429."""
        from src.v2.linkers._errors import RateLimitError
        from src.v2.linkers.s2_author_linker import S2AuthorLinker

        # RateLimitError is raised immediately (not retried), so 1 response needed.
        httpx_mock.add_response(
            url=re.compile(r"https://api\.semanticscholar\.org/graph/v1/author/search.*"),
            status_code=429,
            headers={"Retry-After": "30"},
        )

        linker = S2AuthorLinker(cache_ttl=0)
        with pytest.raises(RateLimitError) as exc_info:
            asyncio.run(linker.match("Alice Smith"))
        assert exc_info.value.source == "s2"


# ─────────────────────────────────────────────────────────────────────────────
# S2AND dedup tests (3 tests) — no HTTP; purely logic + subprocess contract
# ─────────────────────────────────────────────────────────────────────────────


class TestS2ANDDedup:

    def test_empty_input_returns_empty(self):
        """dedup([]) → []"""
        from src.v2.linkers.s2and_dedup import dedup

        result = asyncio.run(dedup([]))
        assert result == []

    def test_graceful_fallback_without_s2and(self, monkeypatch):
        """Without S2AND installed and no S2AND_CLI, returns per-record trivial clusters."""
        from src.v2.linkers import AuthorRecord, dedup

        monkeypatch.delenv("S2AND_CLI", raising=False)

        records = [
            AuthorRecord(block_id="smith_a", author_id="r1", name="Alice Smith"),
            AuthorRecord(block_id="smith_a", author_id="r2", name="A. Smith"),
        ]

        # s2and is not installed — should degrade gracefully, not raise
        clusters = asyncio.run(dedup(records))
        assert len(clusters) == 2
        # trivial path: each record gets its own author_id as cluster
        assert clusters[0] == "r1"
        assert clusters[1] == "r2"

    def test_subprocess_failure_raises_s2and_error(self, monkeypatch, tmp_path):
        """If S2AND_CLI points to a broken script, S2ANDError is raised."""
        from src.v2.linkers import AuthorRecord
        from src.v2.linkers._errors import S2ANDError
        from src.v2.linkers.s2and_dedup import _dedup_subprocess

        # A Python script that exits non-zero
        bad_script_python = tmp_path / "broken_python"
        bad_script_python.write_text("#!/bin/sh\nexit 1\n")
        bad_script_python.chmod(0o755)

        records = [AuthorRecord(block_id="b", author_id="1", name="X")]
        with pytest.raises(S2ANDError):
            _dedup_subprocess(records, str(bad_script_python))


# ─────────────────────────────────────────────────────────────────────────────
# Cache benchmark — same query ≥ 90% faster on second call
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheBenchmark:

    def test_cache_reduces_latency_by_90_percent(self, httpx_mock: HTTPXMock):
        """Second call to RORLinker (same query) hits cache and is ≥90% faster."""
        from src.v2.linkers.ror_linker import RORLinker

        # Only one HTTP response registered — second call must not hit network.
        httpx_mock.add_response(
            url="https://api.ror.org/v2/organizations?affiliation=Stanford+University",
            json=ROR_FIXTURE_MIT,
        )

        linker = RORLinker(cache_ttl=3600)

        t0 = time.perf_counter()
        asyncio.run(linker.find("Stanford University"))
        first_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        asyncio.run(linker.find("Stanford University"))
        second_ms = (time.perf_counter() - t1) * 1000

        # Cache hit should be at least 90% faster (or essentially instant vs network).
        # We accept <10 ms as "essentially instant" to keep the test robust on slow CI.
        assert second_ms < max(first_ms * 0.10, 10.0), (
            f"Cache hit took {second_ms:.1f} ms vs first call {first_ms:.1f} ms — "
            "expected ≥90% speedup"
        )
