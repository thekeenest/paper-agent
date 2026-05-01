"""
PyMuPDF (fitz) parser — raw-text oracle, always available.

Extraction heuristics
---------------------
  headers:           top HEADER_FRAC of page 0, identified by bbox position.
  footnotes:         bottom FOOTNOTE_FRAC of every page, font_size ≤ FOOTNOTE_MAX_SIZE pt.
  acknowledgements:  text blocks between the first "Acknowledgements" section header
                     and the next major section (References / Bibliography / Appendix).
  emails:            regex over each page's raw text; de-duplicated, page-tracked.

No global state; the underlying fitz.Document is opened and closed per call.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from .schemas import Email, Page, ParsedDoc, Span

# ── thresholds ────────────────────────────────────────────────────────────────
HEADER_FRAC: float = 0.35        # top fraction of page-0 counted as "header"
FOOTNOTE_FRAC: float = 0.18      # bottom fraction of any page for footnote detection
FOOTNOTE_MAX_SIZE: float = 9.0   # max font size (pt) to qualify as a footnote span

# ── regexes ──────────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_ACK_HDR_RE = re.compile(r"(?i)^(acknowledgements?|acknowledgments?)\s*$")
_STOP_RE = re.compile(r"(?i)^(references?|bibliography|appendix\b|conclusions?)\s*$")


# ─────────────────────────────── public API ───────────────────────────────────


def parse(pdf_path: Path) -> ParsedDoc:
    """Extract structured content from *pdf_path* using PyMuPDF heuristics.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.  Must exist and be readable.

    Returns
    -------
    ParsedDoc
        Fully populated model; JSON-serialisable.
    """
    doc = fitz.open(str(pdf_path))
    try:
        return _parse_open_doc(doc)
    finally:
        doc.close()


# ─────────────────────────────── internals ────────────────────────────────────


def _parse_open_doc(doc: fitz.Document) -> ParsedDoc:
    pages: list[Page] = []
    raw_page_texts: list[str] = []
    # Collect (page_num, page_width, page_height, raw_blocks) for all pages
    page_data: list[tuple[int, float, float, list[dict]]] = []

    for pnum, page in enumerate(doc):
        pw, ph = page.rect.width, page.rect.height
        raw = page.get_text("text")
        blocks: list[dict] = page.get_text("dict").get("blocks", [])
        pages.append(Page(number=pnum, raw_text=raw, width=pw, height=ph))
        raw_page_texts.append(raw)
        page_data.append((pnum, pw, ph, blocks))

    full_text = "\n\n".join(raw_page_texts)

    return ParsedDoc(
        pages=pages,
        headers=_extract_headers(page_data),
        footnotes=_extract_footnotes(page_data),
        acknowledgements=_extract_acknowledgements(page_data),
        emails=_extract_emails(pages),
        raw_text=full_text,
        parser_name="pymupdf",
    )


def _block_text_and_max_font(block: dict) -> tuple[str, float]:
    """Return (joined_text, max_font_size_pt) for a PyMuPDF text block dict."""
    parts: list[str] = []
    max_fs = 0.0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            t = span.get("text", "").strip()
            if t:
                parts.append(t)
            fs = float(span.get("size", 0))
            if fs > max_fs:
                max_fs = fs
    return " ".join(parts), max_fs


def _extract_headers(
    page_data: list[tuple[int, float, float, list[dict]]],
) -> list[Span]:
    """All text blocks in the top HEADER_FRAC of page 0."""
    if not page_data:
        return []
    pnum, _pw, ph, blocks = page_data[0]
    cutoff = ph * HEADER_FRAC
    spans: list[Span] = []
    for block in blocks:
        if block.get("type") != 0:  # 0 = text block
            continue
        bbox = block.get("bbox")
        if not bbox or bbox[1] >= cutoff:
            continue
        text, max_fs = _block_text_and_max_font(block)
        if not text:
            continue
        confidence = min(1.0, max_fs / 12.0) if max_fs > 0 else 0.5
        spans.append(
            Span(text=text, bbox=tuple(bbox), page=pnum, parser_confidence=confidence)  # type: ignore[arg-type]
        )
    return spans


def _extract_footnotes(
    page_data: list[tuple[int, float, float, list[dict]]],
) -> list[Span]:
    """Small-font blocks in the bottom FOOTNOTE_FRAC of every page."""
    spans: list[Span] = []
    for pnum, _pw, ph, blocks in page_data:
        y_min = ph * (1.0 - FOOTNOTE_FRAC)
        for block in blocks:
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox")
            if not bbox or bbox[1] < y_min:
                continue
            text, max_fs = _block_text_and_max_font(block)
            if not text:
                continue
            # Accept if small font OR no font info (size == 0)
            if max_fs <= FOOTNOTE_MAX_SIZE:
                spans.append(
                    Span(text=text, bbox=tuple(bbox), page=pnum, parser_confidence=0.70)  # type: ignore[arg-type]
                )
    return spans


def _extract_acknowledgements(
    page_data: list[tuple[int, float, float, list[dict]]],
) -> list[Span]:
    """Blocks between the first ACK section header and the next major section."""
    spans: list[Span] = []
    in_ack = False
    for pnum, _pw, _ph, blocks in page_data:
        for block in blocks:
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox")
            text, _ = _block_text_and_max_font(block)
            if not text:
                continue
            stripped = text.strip()
            if in_ack and _STOP_RE.match(stripped):
                return spans  # stop at References / Bibliography
            if _ACK_HDR_RE.match(stripped):
                in_ack = True
                continue  # skip the section header itself
            if in_ack:
                spans.append(
                    Span(
                        text=text,
                        bbox=tuple(bbox) if bbox else None,  # type: ignore[arg-type]
                        page=pnum,
                        parser_confidence=0.75,
                    )
                )
    return spans


def _extract_emails(pages: list[Page]) -> list[Email]:
    """De-duplicated email addresses tracked to their first page of occurrence."""
    seen: set[str] = set()
    emails: list[Email] = []
    for page in pages:
        for m in _EMAIL_RE.finditer(page.raw_text):
            addr = m.group().lower()
            if addr not in seen:
                seen.add(addr)
                emails.append(Email(address=addr, page=page.number))
    return emails
