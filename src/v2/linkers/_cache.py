"""
SQLite-backed response cache for linker HTTP calls.

Each linker uses a named *namespace* (e.g. ``"ror"``, ``"openalex"``).
Entries expire after a per-source TTL.  Cache hits short-circuit all network
calls.

The database is created at ``_CACHE_PATH`` (``linkers/_cache.sqlite`` inside
the package directory) on first use.

Usage
-----
::

    from src.v2.linkers._cache import Cache

    cache = Cache(namespace="ror", ttl_seconds=86_400)  # 24 h

    hit = await cache.get("MIT")
    if hit is None:
        result = await _fetch(...)
        await cache.set("MIT", result)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

# Allow override via env var for testing
_CACHE_PATH = Path(
    os.getenv(
        "LINKER_CACHE_PATH",
        str(Path(__file__).parent / "_cache.sqlite"),
    )
)

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS cache (
    namespace TEXT NOT NULL,
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_CACHE_PATH, check_same_thread=False)
    conn.executescript(_INIT_SQL)
    conn.commit()
    return conn


# Module-level connection — re-created per process, serialised via a lock.
_conn: sqlite3.Connection | None = None
_lock = asyncio.Lock()


def _conn_sync() -> sqlite3.Connection:
    global _conn  # noqa: PLW0603
    if _conn is None:
        _conn = _get_conn()
    return _conn


class Cache:
    """Async-safe SQLite cache for a single namespace."""

    def __init__(self, namespace: str, ttl_seconds: float = 86_400) -> None:
        self.namespace = namespace
        self.ttl = ttl_seconds

    # ── async interface ────────────────────────────────────────────────────

    async def get(self, key: str) -> Any | None:
        """Return deserialised value or ``None`` on miss/expiry."""
        async with _lock:
            return self._get_sync(key)

    async def set(self, key: str, value: Any) -> None:
        """Serialise *value* and store it with a TTL."""
        async with _lock:
            self._set_sync(key, value)

    async def delete(self, key: str) -> None:
        async with _lock:
            _conn_sync().execute(
                "DELETE FROM cache WHERE namespace=? AND key=?",
                (self.namespace, key),
            )
            _conn_sync().commit()

    # ── sync helpers (called while lock is held) ──────────────────────────

    def _get_sync(self, key: str) -> Any | None:
        row = _conn_sync().execute(
            "SELECT value, expires_at FROM cache WHERE namespace=? AND key=?",
            (self.namespace, key),
        ).fetchone()
        if row is None:
            return None
        value_json, expires_at = row
        if time.time() > expires_at:
            _LOG.debug("Cache expired: %s/%s", self.namespace, key)
            _conn_sync().execute(
                "DELETE FROM cache WHERE namespace=? AND key=?",
                (self.namespace, key),
            )
            _conn_sync().commit()
            return None
        return json.loads(value_json)

    def _set_sync(self, key: str, value: Any) -> None:
        _conn_sync().execute(
            """
            INSERT INTO cache (namespace, key, value, expires_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
                value=excluded.value,
                expires_at=excluded.expires_at
            """,
            (self.namespace, key, json.dumps(value, default=str), time.time() + self.ttl),
        )
        _conn_sync().commit()

    # ── housekeeping ──────────────────────────────────────────────────────

    async def purge_expired(self) -> int:
        """Delete all expired rows; returns count deleted."""
        async with _lock:
            cur = _conn_sync().execute(
                "DELETE FROM cache WHERE expires_at < ?", (time.time(),)
            )
            _conn_sync().commit()
            return cur.rowcount

    def get_sync(self, key: str) -> Any | None:
        """Synchronous get — for use in non-async contexts."""
        return self._get_sync(key)

    def set_sync(self, key: str, value: Any) -> None:
        """Synchronous set — for use in non-async contexts."""
        self._set_sync(key, value)
