"""
Nougat (Meta) parser — math/table-aware scientific OCR (optional, heavy).

Activation
----------
  This parser is DISABLED by default.  Set ``NOUGAT=1`` (and optionally
  ``SKIP_HEAVY_PARSERS=0``) in the environment to enable it.

  Without a CUDA-capable GPU the model runs on CPU which is very slow
  (~5–30 min per page). The ensemble always runs correctly without it.

Memory budget: ~2 GB RAM + optional GPU VRAM.

Install::

    pip install paper-agent[nougat]   # adds nougat-ocr + torch

References
----------
  Blecher et al. (2023) "Nougat: Neural Optical Understanding for Academic Documents"
  arXiv:2308.13418
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from .schemas import Email, Page, ParsedDoc, Span

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_ACK_HDR_RE = re.compile(r"(?i)^#+\s*(acknowledgements?|acknowledgments?)\s*$", re.MULTILINE)
_NEXT_HDR_RE = re.compile(r"^#+\s", re.MULTILINE)


def _check_available() -> None:
    """Raise ImportError unless NOUGAT=1 is explicitly set."""
    if not os.getenv("NOUGAT"):
        raise ImportError(
            "Nougat parser is disabled (set NOUGAT=1 to enable). "
            "Note: requires GPU for practical performance."
        )
    if os.getenv("SKIP_HEAVY_PARSERS", "").lower() in ("1", "true", "yes"):
        raise ImportError("SKIP_HEAVY_PARSERS is set — nougat skipped")


@lru_cache(maxsize=1)
def _get_model() -> "NougatModel":  # type: ignore[name-defined]  # noqa: F821
    """Lazy-load and memoize Nougat model weights."""
    import torch  # type: ignore[import]
    from nougat import NougatModel  # type: ignore[import]
    from nougat.utils.checkpoint import get_checkpoint  # type: ignore[import]

    checkpoint = get_checkpoint()
    model = NougatModel.from_pretrained(checkpoint)
    model = model.to(torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


# ─────────────────────────────── public API ───────────────────────────────────


def parse(pdf_path: Path) -> ParsedDoc:
    """Parse *pdf_path* using Nougat's scientific OCR model.

    Raises ``ImportError`` unless ``NOUGAT=1`` is set.
    """
    _check_available()

    import torch  # type: ignore[import]
    from nougat.utils.dataset import LazyDataset  # type: ignore[import]
    from torch.utils.data import DataLoader  # type: ignore[import]

    model = _get_model()

    dataset = LazyDataset(str(pdf_path), model.encoder.prepare_input)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    predictions: list[str] = []
    with torch.no_grad():
        for sample, is_last_page in dataloader:
            model_output = model.inference(image_tensors=sample)
            predictions.extend(model_output["predictions"])

    full_mmd = "\n\n".join(predictions)  # Nougat outputs LaTeX-like mmd format

    headers = _extract_headers_from_mmd(full_mmd)
    footnotes: list[Span] = []  # Nougat encodes footnotes inline — skip for now
    acknowledgements = _extract_acknowledgements_from_mmd(full_mmd)
    emails = _extract_emails(full_mmd)
    pages = [Page(number=i, raw_text=p) for i, p in enumerate(predictions)]

    return ParsedDoc(
        pages=pages,
        headers=headers,
        footnotes=footnotes,
        acknowledgements=acknowledgements,
        emails=emails,
        raw_text=full_mmd,
        parser_name="nougat",
    )


# ─────────────────────────────── helpers ──────────────────────────────────────


def _extract_headers_from_mmd(mmd: str) -> list[Span]:
    """Lines before the first numbered section (\\section{} or # 1. Introduction)."""
    body_re = re.compile(r"^(\\section\{|#\s*\d+[\.\s])", re.M)
    m = body_re.search(mmd)
    pre = mmd[: m.start()] if m else mmd
    spans: list[Span] = []
    for line in pre.splitlines():
        stripped = line.strip()
        if stripped:
            spans.append(Span(text=stripped, page=0, parser_confidence=0.80))
    return spans


def _extract_acknowledgements_from_mmd(mmd: str) -> list[Span]:
    m = _ACK_HDR_RE.search(mmd)
    if not m:
        return []
    after = mmd[m.end():]
    nxt = _NEXT_HDR_RE.search(after)
    body = after[: nxt.start()].strip() if nxt else after.strip()
    return [Span(text=body, page=0, parser_confidence=0.80)] if body else []


def _extract_emails(text: str) -> list[Email]:
    seen: set[str] = set()
    emails: list[Email] = []
    for m in _EMAIL_RE.finditer(text):
        addr = m.group().lower()
        if addr not in seen:
            seen.add(addr)
            emails.append(Email(address=addr, page=0))
    return emails
