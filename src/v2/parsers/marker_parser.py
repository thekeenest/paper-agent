"""
Marker (VikParuchuri/marker-pdf) parser — fast secondary parser.

Marker converts PDFs to structured Markdown using deep-learning layout
detection.  It runs on CPU but is significantly faster than Docling.

Set ``SKIP_HEAVY_PARSERS=1`` to skip model loading in offline environments.

Memory budget: ~1 GB on CPU.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from .schemas import Email, Page, ParsedDoc, Span

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
# Markdown section header that signals acknowledgements
_ACK_HDR_RE = re.compile(r"(?i)^#+\s*(acknowledgements?|acknowledgments?)\s*$", re.MULTILINE)
# Next markdown section header (any level) terminates the ack block
_NEXT_HDR_RE = re.compile(r"^#+\s", re.MULTILINE)


def _check_available() -> None:
    if os.getenv("SKIP_HEAVY_PARSERS", "").lower() in ("1", "true", "yes"):
        raise ImportError("SKIP_HEAVY_PARSERS is set — marker skipped")


@lru_cache(maxsize=1)
def _get_model_dict() -> dict:
    """Lazy-load and memoize Marker's model weights."""
    from marker.models import create_model_dict  # type: ignore[import]

    return create_model_dict()


# ─────────────────────────────── public API ───────────────────────────────────


def parse(pdf_path: Path) -> ParsedDoc:
    """Parse *pdf_path* using Marker's layout-aware Markdown converter.

    Raises ``ImportError`` if marker-pdf is not installed or
    ``SKIP_HEAVY_PARSERS=1`` is set.
    """
    _check_available()

    from marker.converters.pdf import PdfConverter  # type: ignore[import]
    from marker.renderers.markdown import MarkdownRenderer  # type: ignore[import]

    model_dict = _get_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    rendered = converter(str(pdf_path))

    # MarkdownOutput has .markdown attribute
    full_md: str = rendered.markdown if hasattr(rendered, "markdown") else str(rendered)

    headers = _extract_headers_from_md(full_md)
    footnotes = _extract_footnotes_from_md(full_md)
    acknowledgements = _extract_acknowledgements_from_md(full_md)
    emails = _extract_emails_from_md(full_md)

    pages = [Page(number=0, raw_text=full_md)]

    return ParsedDoc(
        pages=pages,
        headers=headers,
        footnotes=footnotes,
        acknowledgements=acknowledgements,
        emails=emails,
        raw_text=full_md,
        parser_name="marker",
    )


# ─────────────────────────────── helpers ──────────────────────────────────────


def _lines_before_first_body_section(md: str) -> list[str]:
    """Return lines up to (not including) the first body section header (##+ Introduction etc.)."""
    body_section = re.compile(r"^#+\s+(abstract|introduction|background|related\s+work)\s*$", re.I | re.M)
    m = body_section.search(md)
    header_block = md[: m.start()] if m else md
    return header_block.splitlines()


def _extract_headers_from_md(md: str) -> list[Span]:
    """Extract title + author block from the pre-abstract markdown region."""
    lines = _lines_before_first_body_section(md)
    spans: list[Span] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip pure markdown artifact lines
        if stripped in ("---", "===", "***"):
            continue
        confidence = 0.90 if stripped.startswith("#") else 0.78
        spans.append(Span(text=stripped.lstrip("#").strip(), page=0, parser_confidence=confidence))
    return spans


def _extract_footnotes_from_md(md: str) -> list[Span]:
    """Heuristic: lines matching common footnote patterns (^[^#] with ¹²³* prefix)."""
    footnote_re = re.compile(r"^\s*[\*†‡§¶\d¹²³⁴⁵⁶⁷⁸⁹⁰]+[.\s].+")
    spans: list[Span] = []
    for line in md.splitlines():
        if footnote_re.match(line) and len(line.strip()) < 200:
            spans.append(Span(text=line.strip(), page=0, parser_confidence=0.65))
    return spans


def _extract_acknowledgements_from_md(md: str) -> list[Span]:
    """Extract acknowledgements section body from markdown."""
    m = _ACK_HDR_RE.search(md)
    if not m:
        return []
    after_ack = md[m.end():]
    next_m = _NEXT_HDR_RE.search(after_ack)
    ack_body = after_ack[: next_m.start()].strip() if next_m else after_ack.strip()
    if not ack_body:
        return []
    return [Span(text=ack_body, page=0, parser_confidence=0.82)]


def _extract_emails_from_md(md: str) -> list[Email]:
    seen: set[str] = set()
    emails: list[Email] = []
    for m in _EMAIL_RE.finditer(md):
        addr = m.group().lower()
        if addr not in seen:
            seen.add(addr)
            emails.append(Email(address=addr, page=0))
    return emails
