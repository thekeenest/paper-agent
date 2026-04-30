# src/v2/parsers/

**Purpose:** Multi-parser PDF extraction ensemble with active disagreement signalling.

| Planned module | Parser | Role |
|---|---|---|
| `docling_parser.py` | Docling (IBM) | Primary — layout-aware, open-source |
| `marker_parser.py` | Marker | Fast secondary |
| `nougat_parser.py` | Nougat | Math/table-aware (optional dep) |
| `pymupdf_parser.py` | PyMuPDF | v1 fallback — always available |
| `ensemble.py` | — | Runs parsers, computes disagreement score, returns `ParseResult` |

Disagreement in the extracted author block triggers a second-pass extraction attempt
via the Planner (see `src/v2/orchestration/planner.py`).

See [DEV_PLAN.md §3.2](../../../coursework_v2/DEV_PLAN.md).
