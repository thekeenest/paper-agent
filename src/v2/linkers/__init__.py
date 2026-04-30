"""
Linker layer — external entity resolution for institutions and authors.

Planned linkers (see DEV_PLAN.md §3.3):
  RORLinker          — /organizations?affiliation= fuzzy search + caching
  OpenAlexLinker     — works/{id} institution resolution; 200M+ works
  S2ANDLinker        — author disambiguation via Semantic Scholar AND model
  EmailDNSResolver   — DNS MX/A lookup for email domains

Each linker returns an EntityLink with a confidence score and source identifier
so the Critic can cross-check extraction candidates.

Status: STUB — no implementation yet.
"""
