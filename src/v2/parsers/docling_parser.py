"""
Docling (IBM) parser — primary layout-aware parser in the ensemble.

Model download
--------------
  First run downloads ~2 GB of model weights from Hugging Face.
  Network access is allowed; subsequent runs use the local cache.
  Set ``SKIP_HEAVY_PARSERS=1`` to raise ImportError before any model load
  (so the ensemble silently skips this parser in offline environments).

Memory budget
-------------
  The Docling pipeline uses ~1–2 GB RAM on CPU; GPU optional.
  The converter is lazily instantiated and memoised via ``_get_converter()``.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from .schemas import Email, Page, ParsedDoc, Span

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_ACK_RE = re.compile(r"(?i)\backnowledgements?\b|\backnowledgments?\b")
_STOP_RE = re.compile(r"(?i)^#+\s*(references?|bibliography|appendix)\b")

# ── optional heavy-parser guard ────────────────────────────────────────────────


def _check_available() -> None:
    if os.getenv("SKIP_HEAVY_PARSERS", "").lower() in ("1", "true", "yes"):
        raise ImportError("SKIP_HEAVY_PARSERS is set — docling skipped")


# ── lazy converter (singleton per process) ─────────────────────────────────────


@lru_cache(maxsize=1)
def _get_converter() -> "DocumentConverter":  # type: ignore[name-defined]  # noqa: F821
    from docling.document_converter import DocumentConverter  # type: ignore[import]

    return DocumentConverter()


# ─────────────────────────────── public API ───────────────────────────────────


def parse(pdf_path: Path) -> ParsedDoc:
    """Parse *pdf_path* using Docling's layout-aware pipeline.

    Raises ``ImportError`` if docling is not installed or
    ``SKIP_HEAVY_PARSERS=1`` is set.
    """
    _check_available()

    from docling_core.types.doc import DocItemLabel  # type: ignore[import]

    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    headers: list[Span] = []
    footnotes: list[Span] = []
    acknowledgements: list[Span] = []
    in_ack = False

    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        text = text.strip()

        label = getattr(item, "label", None)

        # ── provenance → page + bbox ──────────────────────────────────────────
        page_no = 0
        bbox_tuple: tuple[float, float, float, float] | None = None
        prov_list = getattr(item, "prov", [])
        if prov_list:
            prov = prov_list[0]
            page_no = max(0, prov.page_no - 1)  # docling uses 1-indexed pages
            bb = prov.bbox
            bbox_tuple = (float(bb.l), float(bb.t), float(bb.r), float(bb.b))

        # ── section routing ───────────────────────────────────────────────────
        if label == DocItemLabel.SECTION_HEADER:
            if _ACK_RE.search(text):
                in_ack = True
                continue  # skip the header line itself
            if in_ack:
                in_ack = False  # next non-ack section header ends the block

        if in_ack:
            acknowledgements.append(
                Span(text=text, bbox=bbox_tuple, page=page_no, parser_confidence=0.88)
            )
            continue

        if label == DocItemLabel.FOOTNOTE:
            footnotes.append(
                Span(text=text, bbox=bbox_tuple, page=page_no, parser_confidence=0.92)
            )
            continue

        # ── header extraction: title + first-page content ─────────────────────
        if label == DocItemLabel.TITLE:
            headers.append(
                Span(text=text, bbox=bbox_tuple, page=page_no, parser_confidence=0.97)
            )
            continue

        if page_no == 0 and label in (
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
        ):
            headers.append(
                Span(text=text, bbox=bbox_tuple, page=page_no, parser_confidence=0.82)
            )

    # ── email extraction from the full markdown export ─────────────────────────
    full_md: str = doc.export_to_markdown()
    seen_emails: set[str] = set()
    emails: list[Email] = []
    for m in _EMAIL_RE.finditer(full_md):
        addr = m.group().lower()
        if addr not in seen_emails:
            seen_emails.add(addr)
            emails.append(Email(address=addr, page=0))

    pages = [Page(number=0, raw_text=full_md)]

    return ParsedDoc(
        pages=pages,
        headers=headers,
        footnotes=footnotes,
        acknowledgements=acknowledgements,
        emails=emails,
        raw_text=full_md,
        parser_name="docling",
    )
