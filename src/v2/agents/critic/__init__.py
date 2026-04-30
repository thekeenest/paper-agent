"""
Critic / Verifier agent — the methodological core of v2.

The Critic grades extractor candidates by citing external tool evidence:
  - OpenAlex works/{id} lookup (institution strings + ROR IDs)
  - ROR /organizations?affiliation= fuzzy search
  - Semantic Scholar author endpoint
  - Email-domain DNS resolution
  - Raw page-text span match

An accept/reject decision without a cited evidence record is INVALID.
This design follows the CRITIC (Gou et al., 2023) and MAR patterns, specialised
for scholarly affiliation extraction (see DEV_PLAN.md §2.4, READING_LIST.md §A).

Status: STUB — no implementation yet.
"""
