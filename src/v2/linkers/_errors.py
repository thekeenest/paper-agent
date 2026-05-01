"""
Typed exception hierarchy for all v2 linkers.

Usage
-----
::

    from src.v2.linkers._errors import LinkerError, RateLimitError, NotFoundError

    raise RateLimitError("ror", "429 Too Many Requests", retry_after=60)
"""
from __future__ import annotations


class LinkerError(Exception):
    """Base class for all linker errors."""

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        super().__init__(f"[{source}] {message}")


class NetworkError(LinkerError):
    """HTTP or connection-level failure."""


class RateLimitError(LinkerError):
    """Source is throttling us; retry after *retry_after* seconds if known."""

    def __init__(self, source: str, message: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(source, message)


class NotFoundError(LinkerError):
    """Query returned zero results (not an error, but surfaced explicitly)."""


class ParseError(LinkerError):
    """Unexpected response schema — API contract violation."""


class ConfigurationError(LinkerError):
    """Missing env var, bad URL, or install problem."""


class S2ANDError(LinkerError):
    """S2AND-specific errors (subprocess failure, schema mismatch, etc.)."""
