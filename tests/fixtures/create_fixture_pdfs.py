#!/usr/bin/env python3
"""
Generator for the 5 fixture PDFs used by tests/test_parser_ensemble.py.

Run once to (re-)create the fixtures::

    python tests/fixtures/create_fixture_pdfs.py

All PDFs are synthetic, deterministic, and copyright-free.
Content mirrors real academic paper structure so that heuristic parsers
produce meaningful, testable output.

Layout notes (A4 = 595 × 842 pt)
---------------------------------
  HEADER_FRAC = 0.35  →  top cutoff at y = 294.7
  FOOTNOTE_FRAC = 0.18 → bottom cutoff at y = 690.4

Fixture inventory
-----------------
  fixture1_simple.pdf        – clean single-column paper, all regions present
  fixture2_multiaffil.pdf    – multiple affiliations per author, curly-brace emails
  fixture3_footnotes.pdf     – many footnotes, emails inside footnotes
  fixture4_ack_only.pdf      – prominent Acknowledgements section, no footnotes
  fixture5_edge.pdf          – edge case: author block still in header, minimal emails
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

OUT_DIR = Path(__file__).parent / "pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

A4_W, A4_H = 595, 842
L_MARGIN = 72
BODY_FONT = "helv"   # Helvetica
BOLD_FONT = "hebo"   # Helvetica-Bold (fitz built-in alias)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _t(page: fitz.Page, y: float, text: str, size: float = 10, bold: bool = False, x: float = L_MARGIN) -> None:
    """Insert a single line of text."""
    page.insert_text(
        (x, y),
        text,
        fontsize=size,
        fontname=BOLD_FONT if bold else BODY_FONT,
        color=(0, 0, 0),
    )


def _hr(page: fitz.Page, y: float) -> None:
    """Thin horizontal rule."""
    page.draw_line((L_MARGIN, y), (A4_W - L_MARGIN, y), color=(0.5, 0.5, 0.5), width=0.5)


def _body_text(page: fitz.Page, y: float, lines: list[str], size: float = 10) -> float:
    """Insert body lines, return updated y."""
    for line in lines:
        _t(page, y, line, size=size)
        y += size + 4
    return y


# ────────────────────────────────────────────────────────────────────────────
# Fixture 1: simple single-column paper
# ────────────────────────────────────────────────────────────────────────────


def make_fixture1() -> None:
    doc = fitz.open()
    p1 = doc.new_page(width=A4_W, height=A4_H)

    # ── Header region (y < 294.7) ──────────────────────────────────────────
    _t(p1, 60,  "Attention Mechanisms in Neural Networks: A Survey", size=15, bold=True)
    _t(p1, 88,  "Alice Smith\u00b9, Bob Jones\u00b2, Carol Wu\u00b9", size=11)
    _t(p1, 106, "\u00b9MIT CSAIL, Cambridge MA 02139, USA", size=9)
    _t(p1, 120, "\u00b2Stanford NLP Group, Stanford CA 94305, USA", size=9)
    _t(p1, 134, "alice@mit.edu, bjones@stanford.edu, cwu@mit.edu", size=9)
    _hr(p1, 148)
    _t(p1, 160, "Abstract", size=12, bold=True)
    _t(p1, 177, "We survey attention mechanisms in deep learning, covering transformers,", size=10)
    _t(p1, 190, "self-attention, and cross-attention architectures. We analyze 200 papers.", size=10)
    _t(p1, 210, "Keywords: attention, transformers, neural networks, survey", size=9)

    # ── Body (y > 294.7) ──────────────────────────────────────────────────
    y = 310
    _t(p1, y, "1  Introduction", size=12, bold=True);   y += 22
    y = _body_text(p1, y, [
        "Attention mechanisms have become a cornerstone of modern deep learning.",
        "First introduced by Bahdanau et al. (2015) for machine translation, they",
        "have since been adopted across vision, language, and multi-modal tasks.",
        "The transformer architecture (Vaswani et al., 2017) relies exclusively on",
        "attention, dispensing with recurrence entirely.",
    ])
    y += 10
    _t(p1, y, "2  Background", size=12, bold=True);     y += 22
    y = _body_text(p1, y, [
        "Given query Q, keys K, and values V, the scaled dot-product attention is",
        "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V, where d_k is the key dim.",
        "Multi-head attention projects Q, K, V into h parallel heads, concatenates",
        "the outputs, and projects back to model dimension.",
    ])

    # ── Footnotes (y > 690.4) ─────────────────────────────────────────────
    _hr(p1, 700)
    _t(p1, 708, "\u00b9 Corresponding author. Work partially done at Google Brain.", size=7)
    _t(p1, 720, "\u00b2 Now at OpenAI, San Francisco CA 94016.", size=7)

    # ── Page 2 ────────────────────────────────────────────────────────────
    p2 = doc.new_page(width=A4_W, height=A4_H)
    y = 60
    _t(p2, y, "3  Related Work", size=12, bold=True);  y += 22
    y = _body_text(p2, y, [
        "Luong et al. (2015) proposed multiplicative attention as a cheaper alternative.",
        "Transformer-XL (Dai et al., 2019) extends the context window via segment-level",
        "recurrence. BERT (Devlin et al., 2019) uses bidirectional self-attention for",
        "pre-training on masked language modelling.",
    ])
    y += 10
    _t(p2, y, "Acknowledgements", size=12, bold=True);  y += 18
    y = _body_text(p2, y, [
        "The authors thank the MIT CSAIL computing cluster for GPU resources.",
        "This work was supported by NSF grant IIS-2345678 and a Google Faculty Award.",
        "Carol Wu was supported by an MIT Presidential Fellowship.",
    ])
    y += 10
    _t(p2, y, "References", size=12, bold=True);  y += 18
    y = _body_text(p2, y, [
        "[1] Vaswani et al. Attention is all you need. NeurIPS 2017.",
        "[2] Bahdanau et al. Neural machine translation by jointly learning. ICLR 2015.",
        "[3] Devlin et al. BERT: Pre-training of deep bidirectional transformers. 2019.",
    ])

    doc.save(str(OUT_DIR / "fixture1_simple.pdf"))
    doc.close()
    print("fixture1_simple.pdf  ✓")


# ────────────────────────────────────────────────────────────────────────────
# Fixture 2: multiple affiliations, curly-brace email format
# ────────────────────────────────────────────────────────────────────────────


def make_fixture2() -> None:
    doc = fitz.open()
    p1 = doc.new_page(width=A4_W, height=A4_H)

    _t(p1, 60,  "Cross-Lingual Transfer Learning for Low-Resource NLP", size=14, bold=True)
    _t(p1, 85,  "David Lee\u00b9\u00b2, Eva Novak\u00b2\u00b3, Frank Schmidt\u00b9", size=11)
    _t(p1, 103, "\u00b9Technical University of Munich, Germany", size=9)
    _t(p1, 117, "\u00b2Max Planck Institute for Intelligent Systems, Germany", size=9)
    _t(p1, 131, "\u00b3University of Edinburgh, UK", size=9)
    _t(p1, 145, "{dlee, fschmidt}@tum.de  |  e.novak@ed.ac.uk", size=9)
    _hr(p1, 160)
    _t(p1, 172, "Abstract", size=12, bold=True)
    _t(p1, 189, "We present a multilingual pre-training approach that achieves strong", size=10)
    _t(p1, 202, "transfer from high-resource to low-resource languages. Using a shared", size=10)
    _t(p1, 215, "subword vocabulary of 128k tokens, we train on 50 languages simultaneously.", size=10)

    y = 310
    _t(p1, y, "1  Introduction", size=12, bold=True);  y += 22
    y = _body_text(p1, y, [
        "Low-resource languages account for the majority of the world's 7000+ languages",
        "yet receive less than 1% of NLP research attention. Cross-lingual transfer",
        "offers a promising path: pre-train on data-rich languages and fine-tune on",
        "minimal target-language data.",
    ])

    _hr(p1, 700)
    _t(p1, 708, "Correspondence: dlee@tum.de. Code at github.com/tum-nlp/xlt.", size=7)

    p2 = doc.new_page(width=A4_W, height=A4_H)
    y = 60
    _t(p2, y, "Acknowledgements", size=12, bold=True);  y += 18
    y = _body_text(p2, y, [
        "David Lee was funded by DFG grant LE-2345/1-1.",
        "Eva Novak acknowledges support from the MPI-IS International Max Planck",
        "Research School for Intelligent Systems (IMPRS-IS).",
        "Compute was provided by the Leibniz Supercomputing Centre.",
        "Contact: research-group@tum.de",
    ])
    y += 10
    _t(p2, y, "References", size=12, bold=True);  y += 18
    y = _body_text(p2, y, [
        "[1] Conneau et al. Unsupervised cross-lingual representation learning. NeurIPS 2020.",
        "[2] Pires et al. How multilingual is multilingual BERT? ACL 2019.",
    ])

    doc.save(str(OUT_DIR / "fixture2_multiaffil.pdf"))
    doc.close()
    print("fixture2_multiaffil.pdf  ✓")


# ────────────────────────────────────────────────────────────────────────────
# Fixture 3: heavy footnotes (emails embedded in footnotes)
# ────────────────────────────────────────────────────────────────────────────


def make_fixture3() -> None:
    doc = fitz.open()
    p1 = doc.new_page(width=A4_W, height=A4_H)

    _t(p1, 60, "Graph Neural Networks for Molecular Property Prediction", size=13, bold=True)
    _t(p1, 83, "Grace Kim\u00b9\u2217, Hideo Tanaka\u00b2", size=11)
    _t(p1, 100, "\u00b9Caltech, Pasadena CA 91125  |  \u00b2Kyoto University, Japan", size=9)
    _hr(p1, 115)
    _t(p1, 127, "Abstract", size=12, bold=True)
    _t(p1, 144, "Graph neural networks (GNNs) excel at learning molecular representations.", size=10)
    _t(p1, 157, "We benchmark 12 GNN architectures on 8 molecular datasets from MoleculeNet.", size=10)
    _t(p1, 170, "Our proposed MPNNv2 achieves state-of-the-art on 6 of 8 benchmarks.", size=10)

    y = 310
    _t(p1, y, "1  Introduction", size=12, bold=True);  y += 22
    y = _body_text(p1, y, [
        "Predicting molecular properties from graph structure is a fundamental challenge",
        "in computational chemistry and drug discovery. Atoms correspond to nodes;",
        "bonds to edges. A message-passing neural network aggregates neighbourhood",
        "information to produce atom-level and molecule-level representations.",
    ])
    y += 10
    _t(p1, y, "2  Method", size=12, bold=True);  y += 22
    y = _body_text(p1, y, [
        "Our MPNNv2 uses a gated graph convolution [1] with virtual nodes [2].",
        "Edge features (bond type, stereo, aromaticity) are injected into the",
        "message function. We train for 100 epochs with Adam (lr=1e-3).",
    ])

    _hr(p1, 695)
    _t(p1, 703, "\u2217 Corresponding author: gkim@caltech.edu", size=7)
    _t(p1, 714, "\u00b9 Work done while Grace Kim was an intern at DeepMind.", size=7)
    _t(p1, 725, "\u00b2 Hideo Tanaka acknowledges support from JSPS grant 22K21789.", size=7)
    _t(p1, 736, "hideo.tanaka@kyoto-u.ac.jp", size=7)

    p2 = doc.new_page(width=A4_W, height=A4_H)
    y = 60
    _t(p2, y, "3  Experiments", size=12, bold=True);  y += 22
    y = _body_text(p2, y, [
        "We evaluate on HIV, BACE, BBBP, Tox21, ToxCast, SIDER, ClinTox, MUV.",
        "Results show consistent improvement over SchNet, DimeNet, and DimeNet++.",
    ])
    _hr(p2, 695)
    _t(p2, 708, "[1] Gilmer et al. Neural message passing for quantum chemistry. ICML 2017.", size=7)
    _t(p2, 720, "[2] Hu et al. Strategies for pre-training graph neural networks. ICLR 2020.", size=7)

    doc.save(str(OUT_DIR / "fixture3_footnotes.pdf"))
    doc.close()
    print("fixture3_footnotes.pdf  ✓")


# ────────────────────────────────────────────────────────────────────────────
# Fixture 4: prominent acknowledgements, minimal footnotes
# ────────────────────────────────────────────────────────────────────────────


def make_fixture4() -> None:
    doc = fitz.open()
    p1 = doc.new_page(width=A4_W, height=A4_H)

    _t(p1, 60,  "Contrastive Learning for Visual Representations", size=14, bold=True)
    _t(p1, 83,  "Ivan Petrov\u00b9, Julia Chen\u00b2", size=11)
    _t(p1, 100, "\u00b9Moscow State University, Russia", size=9)
    _t(p1, 114, "\u00b2University of Toronto, Canada", size=9)
    _t(p1, 128, "ivanp@msu.ru, julia.chen@utoronto.ca", size=9)
    _hr(p1, 143)
    _t(p1, 155, "Abstract", size=12, bold=True)
    _t(p1, 172, "We propose a contrastive self-supervised learning framework for visual", size=10)
    _t(p1, 185, "representations, achieving 85.4% linear evaluation accuracy on ImageNet", size=10)
    _t(p1, 198, "with a ResNet-50 backbone and no labels during pre-training.", size=10)

    y = 310
    _t(p1, y, "1  Introduction", size=12, bold=True);  y += 22
    y = _body_text(p1, y, [
        "Self-supervised learning has achieved remarkable progress in recent years.",
        "SimCLR (Chen et al., 2020) learns representations by maximising agreement",
        "between augmented views of the same image using a contrastive loss.",
        "MoCo (He et al., 2020) maintains a memory bank of negative keys to enable",
        "large-batch training without a correspondingly large batch size.",
    ])

    p2 = doc.new_page(width=A4_W, height=A4_H)
    y = 60
    _t(p2, y, "2  Experiments", size=12, bold=True);  y += 22
    y = _body_text(p2, y, [
        "We pre-train on ImageNet-1K for 200 epochs using two 180-degree rotations,",
        "colour jitter, and Gaussian blur as augmentations.",
        "Linear evaluation follows the standard protocol from SimCLR.",
    ])
    y += 20
    _t(p2, y, "Acknowledgements", size=12, bold=True);  y += 18
    _t(p2, y, "The authors gratefully acknowledge the Vector Institute, Toronto, for", size=10); y += 15
    _t(p2, y, "providing GPU cluster access. Julia Chen is supported by an NSERC", size=10); y += 15
    _t(p2, y, "Discovery Grant RGPIN-2023-04567 and a Canada CIFAR AI Chair.", size=10); y += 15
    _t(p2, y, "Ivan Petrov was a visiting researcher at the Vector Institute during", size=10); y += 15
    _t(p2, y, "this project; travel was funded by RSF grant 22-21-00741.", size=10); y += 15
    _t(p2, y, "We also thank the anonymous reviewers for their insightful comments.", size=10); y += 25
    _t(p2, y, "References", size=12, bold=True);  y += 18
    y = _body_text(p2, y, [
        "[1] Chen et al. A simple framework for contrastive learning. ICML 2020.",
        "[2] He et al. Momentum contrast for unsupervised visual representation. CVPR 2020.",
        "[3] Grill et al. Bootstrap your own latent. NeurIPS 2020.",
    ])

    doc.save(str(OUT_DIR / "fixture4_ack_only.pdf"))
    doc.close()
    print("fixture4_ack_only.pdf  ✓")


# ────────────────────────────────────────────────────────────────────────────
# Fixture 5: edge case — workshop one-pager, single affiliation, URL-style email
# ────────────────────────────────────────────────────────────────────────────


def make_fixture5() -> None:
    doc = fitz.open()
    p1 = doc.new_page(width=A4_W, height=A4_H)

    _t(p1, 60,  "A Note on Positional Encodings Beyond Sequence Length", size=13, bold=True)
    _t(p1, 82,  "Karl Hoffman", size=11)
    _t(p1, 99,  "ETH Zurich, Switzerland  |  karl.hoffman@inf.ethz.ch", size=9)
    _hr(p1, 113)
    _t(p1, 125, "Abstract", size=12, bold=True)
    _t(p1, 142, "We analyse the behaviour of rotary positional encodings (RoPE) when the", size=10)
    _t(p1, 155, "sequence length at inference exceeds the training context window.", size=10)
    _t(p1, 168, "We derive a closed-form bound on the position-dependent attention decay.", size=10)

    y = 310
    _t(p1, y, "1  Introduction", size=12, bold=True);  y += 22
    y = _body_text(p1, y, [
        "Long-context language models require efficient positional encodings that",
        "extrapolate beyond the training sequence length. Rotary Position Embedding",
        "(RoPE; Su et al., 2022) encodes position as a rotation in the complex plane,",
        "providing relative position information without explicit position embeddings.",
        "However, RoPE's behaviour on out-of-distribution lengths is poorly understood.",
    ])
    y += 10
    _t(p1, y, "2  Analysis", size=12, bold=True);  y += 22
    y = _body_text(p1, y, [
        "Let theta_i = 10000^{-2i/d} for i in {0,...,d/2-1}.",
        "The attention between positions m and n depends on cos((m-n)*theta_i).",
        "For |m-n| > L_train, the cosine oscillates rapidly, reducing effective attention.",
    ])
    y += 10
    _t(p1, y, "Acknowledgements", size=12, bold=True);  y += 18
    _t(p1, y, "This note was written during a research visit to IDSIA, Lugano.", size=10); y += 15
    _t(p1, y, "Funded by SNF Ambizione grant PZ00P2_208900.", size=10); y += 25
    _t(p1, y, "References", size=12, bold=True);  y += 18
    y = _body_text(p1, y, [
        "[1] Su et al. RoFormer: Enhanced transformer with rotary position embedding. 2022.",
        "[2] Press et al. Train short, test long: Attention with linear biases. ICLR 2022.",
    ])

    doc.save(str(OUT_DIR / "fixture5_edge.pdf"))
    doc.close()
    print("fixture5_edge.pdf  ✓")


# ────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    make_fixture1()
    make_fixture2()
    make_fixture3()
    make_fixture4()
    make_fixture5()
    print(f"\nAll fixtures written to {OUT_DIR}/")
