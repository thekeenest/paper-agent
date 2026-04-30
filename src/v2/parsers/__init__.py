"""
Parser ensemble — multi-parser PDF extraction with disagreement signalling.

Planned parsers (see DEV_PLAN.md §3.2):
  DoclingParser   — primary; layout-aware; open-source (IBM, 2024)
  MarkerParser    — fast secondary; open-source (VikParuchuri/marker)
  NougatParser    — math/table-aware; optional extra dependency
  PyMuPDFParser   — v1 fallback; always available

The ensemble detects disagreement in the author block and surfaces it to the
Planner as an active signal (not noise to be smoothed over).

Status: STUB — no implementation yet.
"""
