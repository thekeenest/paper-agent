"""
Shared Pydantic v2 models for the parser ensemble.

All models are frozen and JSON-serialisable so they can be safely passed
between agents and persisted as artefacts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Logical regions of an academic paper that the ensemble inspects.
Region = Literal["header", "footnotes", "acknowledgements", "emails"]


class Span(BaseModel):
    """A contiguous text span extracted from a logical document region."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: tuple[float, float, float, float] | None = None  # (left, top, right, bottom) in points
    page: int
    parser_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Email(BaseModel):
    """An email address found anywhere in the document."""

    model_config = ConfigDict(frozen=True)

    address: str
    author_hint: str | None = None  # nearby text that suggests the owner, if inferable
    page: int
    bbox: tuple[float, float, float, float] | None = None


class Page(BaseModel):
    """Raw content of a single PDF page as seen by one parser."""

    model_config = ConfigDict(frozen=True)

    number: int  # 0-indexed
    raw_text: str
    width: float | None = None   # page width in points
    height: float | None = None  # page height in points


class ParsedDoc(BaseModel):
    """Complete structured extraction from one parser run.

    All list fields are ordered by appearance in the document.
    This model is JSON-serialisable; use .model_dump() / .model_dump_json().
    """

    pages: list[Page]
    headers: list[Span]          # Author-block / title region (top of first page)
    footnotes: list[Span]        # Footnote text (small font at page bottom)
    acknowledgements: list[Span] # Acknowledgements section body
    emails: list[Email]          # All email addresses found in the document
    raw_text: str                # Full concatenated page text
    parser_name: str


# ─────────────────────── disagreement types ───────────────────────────────


class RegionDisagreement(BaseModel):
    """A single mismatch between two parsers in one logical region.

    *similarity* is the best Jaro–Winkler score found when matching
    *span_a* against any span in parser_b's region (0 = no match at all).
    A value below _JW_AGREE_THRESHOLD (0.92) triggers this record.
    """

    region: Region
    parser_a: str
    parser_b: str
    span_a: Span | None = None  # None → span exists only in B
    span_b: Span | None = None  # None → span exists only in A
    similarity: float = Field(ge=0.0, le=1.0)


class DisagreementSet(BaseModel):
    """Full pairwise disagreement report for one PDF across all parsers.

    Usage::

        _, ds = parse_with_ensemble(path)
        header_issues = ds.by_region("header")
        if ds.is_empty:
            print("all parsers agree")
    """

    disagreements: list[RegionDisagreement]
    parsers_used: list[str]
    pdf_path: str
    span_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    # span_counts[parser_name][region] = count of spans returned

    def by_region(self, region: Region) -> list[RegionDisagreement]:
        """Return only the disagreements for *region*."""
        return [d for d in self.disagreements if d.region == region]

    @property
    def is_empty(self) -> bool:
        return not self.disagreements
