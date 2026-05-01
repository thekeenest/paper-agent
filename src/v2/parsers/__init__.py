"""
Parser ensemble — multi-parser PDF extraction with disagreement signalling.

Public surface
--------------
::

    from src.v2.parsers import parse_with_ensemble, ParsedDoc, DisagreementSet

Parsers
-------
  pymupdf   — always available; bbox heuristics (raw-text oracle)
  docling   — primary; IBM layout-aware; set SKIP_HEAVY_PARSERS=1 to skip
  marker    — fast secondary; markdown via Marker; set SKIP_HEAVY_PARSERS=1 to skip
  nougat    — optional; math/table-aware; only active when NOUGAT=1

See DEV_PLAN.md §3.2 and docs/architecture.md for the full design.
"""

from .ensemble import parse_with_ensemble
from .schemas import DisagreementSet, Email, Page, ParsedDoc, Region, RegionDisagreement, Span

__all__ = [
    "parse_with_ensemble",
    "ParsedDoc",
    "DisagreementSet",
    "RegionDisagreement",
    "Span",
    "Email",
    "Page",
    "Region",
]
