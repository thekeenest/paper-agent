"""Tests for src.v2.parsers — PyMuPDF unit tests + ensemble integration tests.

Gold spans are derived from probing the synthetic fixture PDFs with the
PyMuPDF parser (always-available, no heavy deps).

Fixtures
--------
  fixture1_simple.pdf       — single-column, 3 authors, 2 affiliations
  fixture2_multiaffil.pdf   — 3 authors with overlapping dual affiliations
  fixture3_footnotes.pdf    — heavy footnotes (6 spans), no ack section
  fixture4_ack_only.pdf     — explicit Acknowledgements section, no footnotes
  fixture5_edge.pdf         — single author, email embedded in affiliation line
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.v2.parsers import DisagreementSet, ParsedDoc, parse_with_ensemble
from src.v2.parsers.pymupdf_parser import parse as pymupdf_parse
from src.v2.parsers.schemas import Email, Page, Span

FIXTURES = Path(__file__).parent / "fixtures" / "pdfs"

# ─────────────────────────── gold spans (derived from PyMuPDF probe) ─────────

# fixture1_simple.pdf
F1_GOLD_HEADERS = {
    "Attention Mechanisms in Neural Networks: A Survey",
    "Alice Smith¹, Bob Jones², Carol Wu¹",
}
F1_GOLD_EMAILS = {"alice@mit.edu", "bjones@stanford.edu", "cwu@mit.edu"}

# fixture2_multiaffil.pdf
F2_GOLD_HEADERS = {
    "Cross-Lingual Transfer Learning for Low-Resource NLP",
    "David Lee¹², Eva Novak²³, Frank Schmidt¹",
}
F2_GOLD_EMAILS = {"e.novak@ed.ac.uk", "dlee@tum.de"}

# fixture3_footnotes.pdf — 6 footnote spans, 2 emails, no ack section
F3_MIN_FOOTNOTES = 4
F3_GOLD_EMAILS = {"gkim@caltech.edu", "hideo.tanaka@kyoto-u.ac.jp"}

# fixture4_ack_only.pdf
F4_GOLD_EMAILS = {"ivanp@msu.ru", "julia.chen@utoronto.ca"}

# fixture5_edge.pdf — single author, email in affiliation
F5_GOLD_EMAILS = {"karl.hoffman@inf.ethz.ch"}


# ─────────────────────────── recall helpers ──────────────────────────────────


def _header_recall(extracted: list[Span], gold: set[str]) -> float:
    """Fraction of gold strings whose text appears in at least one extracted span."""
    if not gold:
        return 1.0
    hits = sum(
        1 for g in gold if any(g.lower() in s.text.lower() for s in extracted)
    )
    return hits / len(gold)


def _email_recall(extracted: list[Email], gold: set[str]) -> float:
    if not gold:
        return 1.0
    found = {e.address.lower() for e in extracted}
    return len(gold & found) / len(gold)


# ─────────────────────────── PyMuPDF unit tests ──────────────────────────────


class TestPymupdfParser:

    # ── fixture 1 ──

    def test_f1_header_recall(self):
        doc = pymupdf_parse(FIXTURES / "fixture1_simple.pdf")
        assert _header_recall(doc.headers, F1_GOLD_HEADERS) >= 0.9

    def test_f1_email_recall(self):
        doc = pymupdf_parse(FIXTURES / "fixture1_simple.pdf")
        assert _email_recall(doc.emails, F1_GOLD_EMAILS) == 1.0

    def test_f1_footnotes_present(self):
        doc = pymupdf_parse(FIXTURES / "fixture1_simple.pdf")
        assert len(doc.footnotes) >= 1

    def test_f1_acknowledgements_present(self):
        doc = pymupdf_parse(FIXTURES / "fixture1_simple.pdf")
        assert len(doc.acknowledgements) >= 1
        text = doc.acknowledgements[0].text.lower()
        assert "mit" in text or "nsf" in text or "gpu" in text

    def test_f1_parsed_doc_fields(self):
        doc = pymupdf_parse(FIXTURES / "fixture1_simple.pdf")
        assert doc.parser_name == "pymupdf"
        assert len(doc.pages) >= 1
        assert doc.raw_text

    # ── fixture 2 ──

    def test_f2_header_recall(self):
        doc = pymupdf_parse(FIXTURES / "fixture2_multiaffil.pdf")
        assert _header_recall(doc.headers, F2_GOLD_HEADERS) >= 0.9

    def test_f2_email_recall(self):
        doc = pymupdf_parse(FIXTURES / "fixture2_multiaffil.pdf")
        assert _email_recall(doc.emails, F2_GOLD_EMAILS) >= 0.9

    def test_f2_acknowledgements_present(self):
        doc = pymupdf_parse(FIXTURES / "fixture2_multiaffil.pdf")
        assert len(doc.acknowledgements) >= 1

    # ── fixture 3 ──

    def test_f3_heavy_footnotes(self):
        doc = pymupdf_parse(FIXTURES / "fixture3_footnotes.pdf")
        assert len(doc.footnotes) >= F3_MIN_FOOTNOTES

    def test_f3_email_recall(self):
        doc = pymupdf_parse(FIXTURES / "fixture3_footnotes.pdf")
        assert _email_recall(doc.emails, F3_GOLD_EMAILS) == 1.0

    def test_f3_no_acknowledgements(self):
        # fixture3 has no ack section — parser should return empty list
        doc = pymupdf_parse(FIXTURES / "fixture3_footnotes.pdf")
        assert len(doc.acknowledgements) == 0

    # ── fixture 4 ──

    def test_f4_email_recall(self):
        doc = pymupdf_parse(FIXTURES / "fixture4_ack_only.pdf")
        assert _email_recall(doc.emails, F4_GOLD_EMAILS) == 1.0

    def test_f4_acknowledgements_present(self):
        doc = pymupdf_parse(FIXTURES / "fixture4_ack_only.pdf")
        assert len(doc.acknowledgements) >= 1

    def test_f4_no_footnotes(self):
        doc = pymupdf_parse(FIXTURES / "fixture4_ack_only.pdf")
        assert len(doc.footnotes) == 0

    # ── fixture 5 ──

    def test_f5_single_author_headers(self):
        doc = pymupdf_parse(FIXTURES / "fixture5_edge.pdf")
        assert len(doc.headers) >= 3  # title + author line + abstract

    def test_f5_email_in_affiliation_line(self):
        doc = pymupdf_parse(FIXTURES / "fixture5_edge.pdf")
        assert _email_recall(doc.emails, F5_GOLD_EMAILS) == 1.0


# ─────────────────────────── ensemble tests ──────────────────────────────────


class TestEnsemble:

    def test_single_parser_gives_empty_disagreements(self, monkeypatch):
        monkeypatch.setenv("SKIP_HEAVY_PARSERS", "1")
        doc, ds = parse_with_ensemble(
            FIXTURES / "fixture1_simple.pdf",
            parsers=("pymupdf",),
        )
        assert ds.is_empty
        assert ds.parsers_used == ["pymupdf"]

    def test_ensemble_header_recall_f1(self, monkeypatch):
        """With only pymupdf available, ensemble recall on F1 titles ≥ 0.95."""
        monkeypatch.setenv("SKIP_HEAVY_PARSERS", "1")
        doc, _ = parse_with_ensemble(
            FIXTURES / "fixture1_simple.pdf",
            parsers=("docling", "marker", "pymupdf"),
        )
        # docling/marker are skipped; pymupdf is consensus
        assert _header_recall(doc.headers, F1_GOLD_HEADERS) >= 0.95

    def test_ensemble_fallback_to_pymupdf(self, monkeypatch):
        """When all requested parsers are disabled, falls back to pymupdf."""
        monkeypatch.setenv("SKIP_HEAVY_PARSERS", "1")
        doc, _ = parse_with_ensemble(
            FIXTURES / "fixture1_simple.pdf",
            parsers=("docling", "marker"),
        )
        assert doc.parser_name == "pymupdf"

    def test_real_disagreement_two_synthetic_parsers(self, monkeypatch):
        """Inject two synthetic parsers with intentional differences → non-empty DisagreementSet."""

        def _parser_a(path: Path) -> ParsedDoc:
            return ParsedDoc(
                pages=[Page(number=0, raw_text="alpha")],
                headers=[Span(text="Alpha Title: Methods and Results", page=0)],
                footnotes=[],
                acknowledgements=[],
                emails=[Email(address="alpha@example.com", page=0)],
                raw_text="alpha",
                parser_name="parser_a",
            )

        def _parser_b(path: Path) -> ParsedDoc:
            return ParsedDoc(
                pages=[Page(number=0, raw_text="beta")],
                headers=[Span(text="Completely Different Beta Survey", page=0)],
                footnotes=[],
                acknowledgements=[],
                emails=[Email(address="beta@example.org", page=0)],
                raw_text="beta",
                parser_name="parser_b",
            )

        import src.v2.parsers.ensemble as ens_mod

        def _fake_load(name: str):
            return {"parser_a": _parser_a, "parser_b": _parser_b}.get(name)

        monkeypatch.setattr(ens_mod, "_load_parser", _fake_load)

        doc, ds = parse_with_ensemble(
            FIXTURES / "fixture1_simple.pdf",
            parsers=("parser_a", "parser_b"),
        )

        assert not ds.is_empty
        assert set(ds.parsers_used) == {"parser_a", "parser_b"}

        header_disag = ds.by_region("header")
        email_disag = ds.by_region("emails")
        assert len(header_disag) >= 1, "Expected header disagreements"
        assert len(email_disag) >= 1, "Expected email disagreements"
        assert all(d.similarity < 0.92 for d in header_disag)

    def test_disagreement_similarity_range(self, monkeypatch):
        """All RegionDisagreement.similarity values must be in [0, 1]."""
        import src.v2.parsers.ensemble as ens_mod

        def _pa(path):
            return ParsedDoc(
                pages=[Page(number=0, raw_text="x")],
                headers=[Span(text="Foo Bar Baz Title", page=0)],
                footnotes=[Span(text="1. Footnote one.", page=0)],
                acknowledgements=[],
                emails=[],
                raw_text="x",
                parser_name="pa",
            )

        def _pb(path):
            return ParsedDoc(
                pages=[Page(number=0, raw_text="y")],
                headers=[Span(text="Completely Unrelated Document", page=0)],
                footnotes=[],
                acknowledgements=[Span(text="Thanks to nobody.", page=0)],
                emails=[],
                raw_text="y",
                parser_name="pb",
            )

        monkeypatch.setattr(ens_mod, "_load_parser", lambda n: {"pa": _pa, "pb": _pb}.get(n))

        _, ds = parse_with_ensemble(FIXTURES / "fixture1_simple.pdf", parsers=("pa", "pb"))
        for d in ds.disagreements:
            assert 0.0 <= d.similarity <= 1.0

    def test_by_region_filter(self, monkeypatch):
        import src.v2.parsers.ensemble as ens_mod

        def _pa(path):
            return ParsedDoc(
                pages=[Page(number=0, raw_text="x")],
                headers=[Span(text="Title One", page=0)],
                footnotes=[Span(text="Foot note.", page=0)],
                acknowledgements=[],
                emails=[],
                raw_text="x",
                parser_name="pa",
            )

        def _pb(path):
            return ParsedDoc(
                pages=[Page(number=0, raw_text="y")],
                headers=[Span(text="Title Two Entirely Different", page=0)],
                footnotes=[],
                acknowledgements=[],
                emails=[],
                raw_text="y",
                parser_name="pb",
            )

        monkeypatch.setattr(ens_mod, "_load_parser", lambda n: {"pa": _pa, "pb": _pb}.get(n))
        _, ds = parse_with_ensemble(FIXTURES / "fixture1_simple.pdf", parsers=("pa", "pb"))

        # by_region should only return items for that region
        for d in ds.by_region("header"):
            assert d.region == "header"
        for d in ds.by_region("footnotes"):
            assert d.region == "footnotes"

    def test_span_counts_populated(self, monkeypatch):
        monkeypatch.setenv("SKIP_HEAVY_PARSERS", "1")
        _, ds = parse_with_ensemble(
            FIXTURES / "fixture1_simple.pdf",
            parsers=("pymupdf",),
        )
        assert "pymupdf" in ds.span_counts
        counts = ds.span_counts["pymupdf"]
        assert counts["header"] >= 1
        assert isinstance(counts["footnotes"], int)

    def test_all_five_fixtures_parseable(self, monkeypatch):
        monkeypatch.setenv("SKIP_HEAVY_PARSERS", "1")
        for fname in [
            "fixture1_simple.pdf",
            "fixture2_multiaffil.pdf",
            "fixture3_footnotes.pdf",
            "fixture4_ack_only.pdf",
            "fixture5_edge.pdf",
        ]:
            doc, ds = parse_with_ensemble(FIXTURES / fname, parsers=("pymupdf",))
            assert doc.parser_name == "pymupdf"
            assert len(doc.pages) >= 1
            assert ds.parsers_used == ["pymupdf"]

    def test_pdf_path_recorded_in_disagreement_set(self, monkeypatch):
        monkeypatch.setenv("SKIP_HEAVY_PARSERS", "1")
        pdf = FIXTURES / "fixture1_simple.pdf"
        _, ds = parse_with_ensemble(pdf, parsers=("pymupdf",))
        assert str(pdf) == ds.pdf_path


# ─────────────────────────── CLI smoke tests ─────────────────────────────────


class TestCLI:

    def test_cli_returns_valid_json(self):
        env = os.environ.copy()
        env["SKIP_HEAVY_PARSERS"] = "1"
        result = subprocess.run(
            [
                sys.executable, "-m", "src.v2.parsers.ensemble",
                str(FIXTURES / "fixture1_simple.pdf"),
                "pymupdf",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        data = json.loads(result.stdout)
        assert data["consensus_parser"] == "pymupdf"
        assert len(data["headers"]) >= 1
        assert len(data["emails"]) >= 1
        assert "disagreements" in data

    def test_cli_missing_args_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.v2.parsers.ensemble"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_cli_all_five_fixtures(self):
        env = os.environ.copy()
        env["SKIP_HEAVY_PARSERS"] = "1"
        for fname in [
            "fixture1_simple.pdf",
            "fixture2_multiaffil.pdf",
            "fixture3_footnotes.pdf",
            "fixture4_ack_only.pdf",
            "fixture5_edge.pdf",
        ]:
            result = subprocess.run(
                [
                    sys.executable, "-m", "src.v2.parsers.ensemble",
                    str(FIXTURES / fname),
                    "pymupdf",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0, f"{fname}: {result.stderr}"
            data = json.loads(result.stdout)
            assert data["consensus_parser"] == "pymupdf"


# ─────────────────────────── Nougat graceful skip ────────────────────────────


class TestNougatGracefulSkip:

    def test_nougat_not_loaded_without_env(self, monkeypatch):
        """_load_parser('nougat') returns None when NOUGAT env var is absent."""
        monkeypatch.delenv("NOUGAT", raising=False)
        from src.v2.parsers.ensemble import _load_parser
        assert _load_parser("nougat") is None

    def test_nougat_skipped_in_ensemble(self, monkeypatch):
        """Ensemble skips nougat and falls through to pymupdf."""
        monkeypatch.delenv("NOUGAT", raising=False)
        monkeypatch.setenv("SKIP_HEAVY_PARSERS", "1")
        doc, ds = parse_with_ensemble(
            FIXTURES / "fixture1_simple.pdf",
            parsers=("nougat", "pymupdf"),
        )
        assert doc.parser_name == "pymupdf"
        assert "nougat" not in ds.parsers_used

    def test_nougat_not_installed_with_env_still_graceful(self, monkeypatch):
        """With NOUGAT=1 but nougat package absent, _load_parser returns None."""
        monkeypatch.setenv("NOUGAT", "1")
        from src.v2.parsers.ensemble import _load_parser
        fn = _load_parser("nougat")
        # Either None (not installed → ImportError caught) or callable (installed) — both OK
        assert fn is None or callable(fn)

    def test_nougat_importerror_in_nougat_parser_module(self, monkeypatch):
        """nougat_parser.parse() raises ImportError when NOUGAT env not set."""
        monkeypatch.delenv("NOUGAT", raising=False)
        from src.v2.parsers import nougat_parser
        with pytest.raises(ImportError, match="NOUGAT"):
            nougat_parser.parse(FIXTURES / "fixture1_simple.pdf")
