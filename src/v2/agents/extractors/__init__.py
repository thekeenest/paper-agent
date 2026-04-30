"""
Specialist extractor agents.

Planned agents (see DEV_PLAN.md §3.3):
  HeaderExtractor        — parses author block from PDF header region
  FootnoteExtractor      — parses affiliation footnotes / superscript markers
  EmailDomainExtractor   — infers institution from author email domains
  AcknowledgementsAgent  — extracts secondary affiliations from acknowledgements

Each extractor returns a typed ExtractionCandidate with a confidence score and
a source_region field so the Critic can locate the evidence in the PDF.

Status: STUB — no implementation yet.
"""
