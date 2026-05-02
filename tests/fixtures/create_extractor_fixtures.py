"""
Generate 10 synthetic fixture PDFs for extractor tests.

Each covers a different academic paper template pattern:
  fx01 — NeurIPS style (footnote affiliations, long author list)
  fx02 — ICML style (inline affiliations, acknowledgements)
  fx03 — CVPR style (compact header, many footnotes)
  fx04 — ACL style (author + affiliation on separate lines)
  fx05 — Single author, no footnotes
  fx06 — Large lab collaboration (>6 authors)
  fx07 — Industry lab (Google / Meta / DeepMind)
  fx08 — Mixed academia + industry
  fx09 — Non-ASCII author names (Chinese, German umlauts)
  fx10 — Edge case: email only, no affiliation text

Run:  python tests/fixtures/create_extractor_fixtures.py
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

OUT_DIR = Path(__file__).parent / "pdfs"
OUT_DIR.mkdir(exist_ok=True)

BODY_FONT = "helv"
BOLD_FONT = "hebo"
A4_W, A4_H = 595, 842
MARGIN = 60
LINE_H = 14


def _page(doc: fitz.Document) -> fitz.Page:
    return doc.new_page(width=A4_W, height=A4_H)


def _line(page: fitz.Page, y: float, text: str, size: float = 11, bold: bool = False, color: tuple = (0, 0, 0)) -> float:
    font = BOLD_FONT if bold else BODY_FONT
    page.insert_text((MARGIN, y), text, fontname=font, fontsize=size, color=color)
    return y + LINE_H * (size / 11)


def _para(page: fitz.Page, y: float, text: str, size: float = 10, width: int = 475) -> float:
    rc = fitz.Rect(MARGIN, y, MARGIN + width, y + 200)
    page.insert_textbox(rc, text, fontname=BODY_FONT, fontsize=size)
    lines = (len(text) // 90) + 1
    return y + lines * LINE_H * (size / 11) + 4


def _footnote_line(page: fitz.Page, y: float, text: str) -> float:
    page.insert_text((MARGIN, y), text, fontname=BODY_FONT, fontsize=8, color=(0.3, 0.3, 0.3))
    return y + 10


ABSTRACT = (
    "We present a novel approach combining attention mechanisms with graph neural "
    "networks for improved molecular property prediction. Experiments on standard "
    "benchmarks demonstrate state-of-the-art performance across multiple datasets."
)

ACK_TEXTS = {
    "ack_nsf": (
        "Alice Smith is supported by NSF grant IIS-2012345. Bob Jones acknowledges "
        "support from the EPSRC Doctoral Training Partnership (DTP). The authors "
        "thank the MIT CSAIL computing cluster for GPU resources."
    ),
    "ack_dfg": (
        "David Lee was funded by DFG grant LE-2345/1-1. The authors thank the "
        "Max Planck Institute for providing compute resources."
    ),
    "ack_google": (
        "This work was supported in part by a Google Research Award. Grace Kim "
        "is a Google PhD Fellow. Hideo Tanaka is supported by JSPS grant 22K21789."
    ),
}

INTRO_TEXT = (
    "1 Introduction\n\n"
    "Deep learning has transformed many areas of scientific computing. In this "
    "paper we propose a unified framework that addresses key limitations of "
    "prior work. Our contributions are: (1) a novel architecture, (2) improved "
    "training procedure, (3) comprehensive benchmarks."
)


# ─────────────────────────── fx01: NeurIPS style ─────────────────────────────

def make_fx01() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "Attention-Enhanced Graph Networks for Molecular Property Prediction", 14, bold=True)
    y += 6
    y = _line(p, y, "Alice Smith¹², Bob Jones², Carol Wu¹, Dan Brown³", 11)
    y += 4
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    # Footer / footnotes
    yf = 760
    yf = _footnote_line(p, yf, "¹ MIT CSAIL, Cambridge MA 02139, USA")
    yf = _footnote_line(p, yf, "² Stanford NLP Group, Stanford CA 94305, USA")
    yf = _footnote_line(p, yf, "³ Carnegie Mellon University, Pittsburgh PA 15213, USA")
    yf = _footnote_line(p, yf, "Corresponding author: alice@mit.edu")

    # Page 2
    p2 = _page(doc)
    y2 = 80
    y2 = _line(p2, y2, "Acknowledgements", 12, bold=True)
    y2 = _para(p2, y2, ACK_TEXTS["ack_nsf"])

    doc.save(str(OUT_DIR / "fx01_neurips.pdf"))
    doc.close()


# ─────────────────────────── fx02: ICML style ────────────────────────────────

def make_fx02() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "Cross-Lingual Transfer for Low-Resource NLP via Shared Representations", 14, bold=True)
    y += 6
    y = _line(p, y, "David Lee", 12, bold=True)
    y = _line(p, y, "Technical University of Munich, Germany", 10)
    y = _line(p, y, "dlee@tum.de", 10)
    y += 6
    y = _line(p, y, "Eva Novak", 12, bold=True)
    y = _line(p, y, "Max Planck Institute for Intelligent Systems, Germany", 10)
    y = _line(p, y, "e.novak@tuebingen.mpg.de", 10)
    y += 8
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    p2 = _page(doc)
    y2 = 80
    y2 = _line(p2, y2, "Acknowledgements", 12, bold=True)
    y2 = _para(p2, y2, ACK_TEXTS["ack_dfg"])

    doc.save(str(OUT_DIR / "fx02_icml.pdf"))
    doc.close()


# ─────────────────────────── fx03: CVPR style ────────────────────────────────

def make_fx03() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "ViT-GNN: Vision Transformers Meet Graph Neural Networks", 14, bold=True)
    y += 4
    y = _line(p, y, "Grace Kim·  Hideo Tanaka†  Frank Schmidt‡", 11)
    y += 4
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    yf = 720
    yf = _footnote_line(p, yf, "· Grace Kim — Caltech, Pasadena CA 91125; gkim@caltech.edu")
    yf = _footnote_line(p, yf, "† Hideo Tanaka — Kyoto University, Japan; hideo.tanaka@kyoto-u.ac.jp")
    yf = _footnote_line(p, yf, "‡ Frank Schmidt — Technical University of Munich; fschmidt@tum.de")
    yf = _footnote_line(p, yf, "Work done while Grace Kim was an intern at Google Research.")
    yf = _footnote_line(p, yf, "Hideo Tanaka acknowledges support from JSPS grant 22K21789.")

    p2 = _page(doc)
    y2 = 80
    y2 = _line(p2, y2, "Acknowledgements", 12, bold=True)
    y2 = _para(p2, y2, ACK_TEXTS["ack_google"])

    doc.save(str(OUT_DIR / "fx03_cvpr.pdf"))
    doc.close()


# ─────────────────────────── fx04: ACL style ─────────────────────────────────

def make_fx04() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "Multilingual Named Entity Recognition with Subword Encodings", 14, bold=True)
    y += 4
    y = _line(p, y, "Ivan Petrov", 11, bold=True)
    y = _line(p, y, "Moscow State University", 10)
    y = _line(p, y, "ivanp@msu.ru", 10)
    y += 6
    y = _line(p, y, "Julia Chen", 11, bold=True)
    y = _line(p, y, "University of Toronto", 10)
    y = _line(p, y, "julia.chen@utoronto.ca", 10)
    y += 8
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    doc.save(str(OUT_DIR / "fx04_acl.pdf"))
    doc.close()


# ─────────────────────────── fx05: Single author ─────────────────────────────

def make_fx05() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "A Note on Positional Encodings Beyond Training Context Window", 14, bold=True)
    y += 4
    y = _line(p, y, "Karl Hoffman", 12, bold=True)
    y = _line(p, y, "ETH Zurich, Switzerland  |  karl.hoffman@inf.ethz.ch", 10)
    y += 8
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    doc.save(str(OUT_DIR / "fx05_single.pdf"))
    doc.close()


# ─────────────────────────── fx06: Large collaboration ───────────────────────

def make_fx06() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "Scaling Laws for Foundation Models in Scientific Discovery", 14, bold=True)
    y += 4
    y = _line(p, y, "Alice Smith¹, Bob Jones², Carol Wu¹, Dan Brown³, Eva Novak⁴, Frank Lee⁵, Grace Kim¹", 10)
    y += 4
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    yf = 700
    yf = _footnote_line(p, yf, "¹ MIT CSAIL, Cambridge MA 02139, USA   {alice, carol, grace}@mit.edu")
    yf = _footnote_line(p, yf, "² Stanford University, Stanford CA 94305, USA   bjones@stanford.edu")
    yf = _footnote_line(p, yf, "³ Carnegie Mellon University, Pittsburgh PA 15213, USA   dbrown@cs.cmu.edu")
    yf = _footnote_line(p, yf, "⁴ Max Planck Institute, Tübingen, Germany   e.novak@tuebingen.mpg.de")
    yf = _footnote_line(p, yf, "⁵ University of Edinburgh, UK   frank.lee@ed.ac.uk")

    doc.save(str(OUT_DIR / "fx06_large.pdf"))
    doc.close()


# ─────────────────────────── fx07: Industry lab ──────────────────────────────

def make_fx07() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "Gemini: A Family of Highly Capable Multimodal Models (Excerpt)", 14, bold=True)
    y += 4
    y = _line(p, y, "Rohan Anil  Sebastien Borgeaud  Yonghui Wu", 11)
    y = _line(p, y, "Google DeepMind, London, UK", 10)
    y = _line(p, y, "{ranil, sborgeaud, ywu}@google.com", 10)
    y += 8
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    doc.save(str(OUT_DIR / "fx07_industry.pdf"))
    doc.close()


# ─────────────────────────── fx08: Mixed academia + industry ─────────────────

def make_fx08() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "Open-Source LLM Benchmarking: A Community Effort", 14, bold=True)
    y += 4
    y = _line(p, y, "Tom Brown¹²*, Jane Doe²", 11)
    y += 4
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    yf = 740
    yf = _footnote_line(p, yf, "¹ Princeton University, Princeton NJ 08544, USA")
    yf = _footnote_line(p, yf, "² OpenAI, San Francisco CA 94016, USA")
    yf = _footnote_line(p, yf, "* Correspondence: tbrown@openai.com")

    doc.save(str(OUT_DIR / "fx08_mixed.pdf"))
    doc.close()


# ─────────────────────────── fx09: Non-ASCII names ───────────────────────────

def make_fx09() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "Contrastive Learning for Cross-Modal Retrieval", 14, bold=True)
    y += 4
    y = _line(p, y, "Wei Zhang¹, Müller Jörg², Carlos García³", 11)
    y += 4
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    yf = 750
    yf = _footnote_line(p, yf, "¹ Tsinghua University, Beijing 100084, China   wzhang@tsinghua.edu.cn")
    yf = _footnote_line(p, yf, "² Technical University of Munich, Germany   joerg.mueller@tum.de")
    yf = _footnote_line(p, yf, "³ Universidad Autónoma de Madrid, Spain   cgarcia@uam.es")

    doc.save(str(OUT_DIR / "fx09_nonascii.pdf"))
    doc.close()


# ─────────────────────────── fx10: Email-only header ─────────────────────────

def make_fx10() -> None:
    doc = fitz.open()
    p = _page(doc)
    y = 80
    y = _line(p, y, "Dense Retrieval Augmented Generation for Open-Domain QA", 14, bold=True)
    y += 4
    y = _line(p, y, "Maria Santos, Lukas Fischer", 11)
    y = _line(p, y, "m.santos@uva.nl, lukas.fischer@uzh.ch", 10)
    y += 8
    y = _line(p, y, "Abstract", 11, bold=True)
    y = _para(p, y, ABSTRACT)
    y = _para(p, y, INTRO_TEXT)

    # No footnotes at all, affiliations only derivable from emails
    doc.save(str(OUT_DIR / "fx10_email_only.pdf"))
    doc.close()


# ─────────────────────────── runner ──────────────────────────────────────────

if __name__ == "__main__":
    funcs = [make_fx01, make_fx02, make_fx03, make_fx04, make_fx05,
             make_fx06, make_fx07, make_fx08, make_fx09, make_fx10]
    for fn in funcs:
        fn()
        print(f"Created {fn.__name__.replace('make_', '')}.pdf")
    print(f"\nAll 10 fixture PDFs written to {OUT_DIR}")
