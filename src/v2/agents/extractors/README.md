# src/v2/agents/extractors/

**Purpose:** Specialist extractor agents — each agent targets one region of the PDF.

| Planned module | Region targeted |
|---|---|
| `header.py` | Author block in paper header |
| `footnote.py` | Affiliation footnotes with superscript markers |
| `email_domain.py` | Institution inferred from author email domains |
| `acknowledgements.py` | Secondary affiliations in the Acknowledgements section |

All extractors return `ExtractionCandidate` with `confidence` and `source_region`.
The Critic in `src/v2/agents/critic/` grades candidates against external evidence.

See [DEV_PLAN.md §3.3](../../../../coursework_v2/DEV_PLAN.md).
