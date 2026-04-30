"""
Knowledge-graph layer — KuzuDB schema, ingestion, and GraphRAG queries.

Nodes: Author, Paper, Institution, Venue, Topic, Evidence
Edges: AUTHORED, AFFILIATED_AT, PUBLISHED_AT, ABOUT, CHILD_OF,
       COAUTHORED_WITH, COLLABORATED_WITH

Evidence nodes are first-class citizens; every extracted fact is anchored to
an Evidence node recording source, raw_payload, and retrieved_at.

Planned modules (see DEV_PLAN.md §3.5):
  schema.py    — Kuzu DDL: CREATE NODE/REL TABLE statements
  ingest.py    — idempotent upsert pipeline from ExtractionResult → KG
  queries.py   — Cypher query library (top-institutions, collaboration trends)
  graphrag.py  — natural-language Q&A over KG via LangChain GraphRAG

Status: STUB — no implementation yet.
"""
