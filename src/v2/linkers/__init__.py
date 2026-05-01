"""
Linker layer — external entity resolution for institutions and authors.

Public surface
--------------
::

    from src.v2.linkers import (
        RORLinker, ROR_Match,
        OpenAlexInstitutions, Institution, Authorship,
        S2AuthorLinker, S2Author,
        dedup, AuthorRecord, ClusterId,
        LinkerError, NetworkError, RateLimitError, NotFoundError,
        ParseError, ConfigurationError, S2ANDError,
    )

Linkers
-------
  RORLinker              — /organizations?affiliation= fuzzy search
  OpenAlexInstitutions   — by_ror_id / search / by_paper
  S2AuthorLinker         — author name → S2 author records
  dedup                  — S2AND author disambiguation

Each linker:
  * is async-first (usable synchronously via ``asyncio.run``)
  * caches responses in ``linkers/_cache.sqlite`` with per-source TTL
  * raises typed ``LinkerError`` sub-classes (no silent failures)
  * attaches ``provenance`` and ``retrieved_at`` to every result model

See DEV_PLAN.md §3.3 and docs/architecture.md for the full design.
"""

from ._errors import (
    ConfigurationError,
    LinkerError,
    NetworkError,
    NotFoundError,
    ParseError,
    RateLimitError,
    S2ANDError,
)
from .openalex_institutions import Authorship, Institution, OpenAlexInstitutions
from .ror_linker import ROR_Match, RORLinker
from .s2_author_linker import S2Author, S2AuthorLinker
from .s2and_dedup import AuthorRecord, ClusterId, dedup

__all__ = [
    # ROR
    "RORLinker",
    "ROR_Match",
    # OpenAlex
    "OpenAlexInstitutions",
    "Institution",
    "Authorship",
    # S2 author
    "S2AuthorLinker",
    "S2Author",
    # S2AND dedup
    "dedup",
    "AuthorRecord",
    "ClusterId",
    # errors
    "LinkerError",
    "NetworkError",
    "RateLimitError",
    "NotFoundError",
    "ParseError",
    "ConfigurationError",
    "S2ANDError",
]
