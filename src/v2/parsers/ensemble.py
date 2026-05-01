"""
Parser ensemble — runs multiple parsers and surfaces disagreements.

Public API
----------
::

    from src.v2.parsers.ensemble import parse_with_ensemble

    consensus, ds = parse_with_ensemble(
        Path("paper.pdf"),
        parsers=("docling", "marker", "pymupdf"),
    )
    for d in ds.by_region("header"):
        print(d.parser_a, "vs", d.parser_b, ":", d.similarity)

CLI
---
::

    python -m src.v2.parsers.ensemble paper.pdf [parser1 parser2 ...]

Disagreement definition
-----------------------
Two spans *agree* iff their normalised Jaro–Winkler similarity ≥ 0.92.
A span from parser A is a *disagreement* if its best match in parser B
has similarity < 0.92.  Spans present in one parser but missing entirely
from another are also recorded (similarity = 0.0).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from rapidfuzz.distance import JaroWinkler  # type: ignore[import]

from .schemas import DisagreementSet, Email, ParsedDoc, Region, RegionDisagreement, Span

_LOG = logging.getLogger(__name__)

_REGIONS: tuple[Region, ...] = ("header", "footnotes", "acknowledgements", "emails")
_JW_AGREE: float = 0.92  # Jaro–Winkler threshold for "agreement"


# ─────────────────────────────── public API ───────────────────────────────────


def parse_with_ensemble(
    pdf_path: Path,
    parsers: tuple[str, ...] = ("docling", "marker", "pymupdf"),
) -> tuple[ParsedDoc, DisagreementSet]:
    """Run *parsers* on *pdf_path* and return ``(consensus, disagreements)``.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    parsers:
        Names of parsers to try, in priority order.  Available names:
        ``"docling"``, ``"marker"``, ``"pymupdf"``, ``"nougat"``.
        Parsers that are not installed or disabled by env var are silently
        skipped.  If *no* requested parser is available, PyMuPDF is used as
        an absolute fallback.

    Returns
    -------
    consensus:
        The ``ParsedDoc`` from the highest-priority parser that succeeded.
    disagreements:
        Pairwise comparison across all parsers that ran.  Empty when only
        one parser succeeded.
    """
    pdf_path = Path(pdf_path)
    results: dict[str, ParsedDoc] = {}

    for name in parsers:
        fn = _load_parser(name)
        if fn is None:
            _LOG.debug("Parser %r unavailable — skipping", name)
            continue
        try:
            results[name] = fn(pdf_path)
            _LOG.debug("Parser %r: headers=%d emails=%d",
                        name, len(results[name].headers), len(results[name].emails))
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Parser %r failed on %s: %s", name, pdf_path.name, exc)

    # Absolute fallback — PyMuPDF is always available
    if not results:
        _LOG.warning("No requested parser succeeded; falling back to pymupdf")
        from . import pymupdf_parser
        results["pymupdf"] = pymupdf_parser.parse(pdf_path)

    parsers_used = list(results.keys())
    consensus = results[parsers_used[0]]  # first-priority parser wins

    if len(results) < 2:
        ds = DisagreementSet(
            disagreements=[],
            parsers_used=parsers_used,
            pdf_path=str(pdf_path),
            span_counts=_count_spans(results),
        )
    else:
        ds = _build_disagreement_set(results, str(pdf_path))

    return consensus, ds


# ─────────────────────────── parser loader ────────────────────────────────────


def _load_parser(name: str) -> Callable[[Path], ParsedDoc] | None:
    """Return the ``parse()`` callable for *name*, or ``None`` if unavailable."""
    skip_heavy = os.getenv("SKIP_HEAVY_PARSERS", "").lower() in ("1", "true", "yes")

    try:
        if name == "pymupdf":
            from . import pymupdf_parser
            return pymupdf_parser.parse

        if name == "docling":
            if skip_heavy:
                return None
            from . import docling_parser
            return docling_parser.parse

        if name == "marker":
            if skip_heavy:
                return None
            from . import marker_parser
            return marker_parser.parse

        if name == "nougat":
            if not os.getenv("NOUGAT"):
                return None
            from . import nougat_parser
            return nougat_parser.parse

        _LOG.warning("Unknown parser name %r — ignored", name)
        return None

    except ImportError as exc:
        _LOG.debug("Parser %r not importable: %s", name, exc)
        return None


# ─────────────────────────── disagreement logic ───────────────────────────────


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace, for Jaro–Winkler comparison."""
    return " ".join(text.lower().split())


def _jw(a: str, b: str) -> float:
    return float(JaroWinkler.similarity(_normalize(a), _normalize(b)))


def _spans_for_region(doc: ParsedDoc, region: Region) -> list[Span]:
    """Extract Span objects for *region* from *doc* (emails are converted to Spans)."""
    if region == "header":
        return list(doc.headers)
    if region == "footnotes":
        return list(doc.footnotes)
    if region == "acknowledgements":
        return list(doc.acknowledgements)
    if region == "emails":
        return [Span(text=e.address, page=e.page, bbox=e.bbox) for e in doc.emails]
    return []


def _compare_two(
    spans_a: list[Span],
    spans_b: list[Span],
    region: Region,
    name_a: str,
    name_b: str,
) -> list[RegionDisagreement]:
    """Return disagreements between *spans_a* (from *name_a*) and *spans_b*."""
    out: list[RegionDisagreement] = []

    if not spans_a and not spans_b:
        return out

    if not spans_b:
        for sa in spans_a:
            out.append(RegionDisagreement(
                region=region, parser_a=name_a, parser_b=name_b,
                span_a=sa, span_b=None, similarity=0.0,
            ))
        return out

    if not spans_a:
        for sb in spans_b:
            out.append(RegionDisagreement(
                region=region, parser_a=name_a, parser_b=name_b,
                span_a=None, span_b=sb, similarity=0.0,
            ))
        return out

    # For each span in A, find the best match in B.
    matched_b_indices: set[int] = set()
    for sa in spans_a:
        sims = [(_jw(sa.text, sb.text), j) for j, sb in enumerate(spans_b)]
        best_sim, best_j = max(sims)
        if best_sim >= _JW_AGREE:
            matched_b_indices.add(best_j)
        else:
            out.append(RegionDisagreement(
                region=region, parser_a=name_a, parser_b=name_b,
                span_a=sa, span_b=spans_b[best_j], similarity=best_sim,
            ))

    # Spans in B that were never the best match for any A span.
    for j, sb in enumerate(spans_b):
        if j in matched_b_indices:
            continue
        # Confirm no A span reached agreement with this B span either.
        best_from_a = max((_jw(sa.text, sb.text) for sa in spans_a), default=0.0)
        if best_from_a < _JW_AGREE:
            out.append(RegionDisagreement(
                region=region, parser_a=name_a, parser_b=name_b,
                span_a=None, span_b=sb, similarity=best_from_a,
            ))
    return out


def _count_spans(results: dict[str, ParsedDoc]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "header": len(doc.headers),
            "footnotes": len(doc.footnotes),
            "acknowledgements": len(doc.acknowledgements),
            "emails": len(doc.emails),
        }
        for name, doc in results.items()
    }


def _build_disagreement_set(results: dict[str, ParsedDoc], pdf_path: str) -> DisagreementSet:
    names = list(results.keys())
    all_disagreements: list[RegionDisagreement] = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            for region in _REGIONS:
                sa = _spans_for_region(results[na], region)
                sb = _spans_for_region(results[nb], region)
                all_disagreements.extend(_compare_two(sa, sb, region, na, nb))

    return DisagreementSet(
        disagreements=all_disagreements,
        parsers_used=names,
        pdf_path=pdf_path,
        span_counts=_count_spans(results),
    )


# ─────────────────────────────────── CLI ──────────────────────────────────────


def _cli() -> None:
    """Entry point for ``python -m src.v2.parsers.ensemble <pdf> [parsers...]``."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.v2.parsers.ensemble <pdf_path> [parser1 parser2 ...]",
              file=sys.stderr)
        sys.exit(1)

    pdf = Path(sys.argv[1])
    requested = tuple(sys.argv[2:]) if len(sys.argv) > 2 else ("docling", "marker", "pymupdf")

    consensus, ds = parse_with_ensemble(pdf, parsers=requested)

    output = {
        "pdf": str(pdf),
        "consensus_parser": consensus.parser_name,
        "headers": [s.model_dump() for s in consensus.headers],
        "emails": [e.model_dump() for e in consensus.emails],
        "footnotes": [s.model_dump() for s in consensus.footnotes],
        "acknowledgements": [s.model_dump() for s in consensus.acknowledgements],
        "disagreements": ds.model_dump(),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
