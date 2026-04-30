"""
Paper-Agent v2 — public interface.

All new code lives under src/v2/. Import from here inside v2 modules
to avoid deep cross-package imports:

    from src.v2 import CoordinatorState, ExtractionResult

Sub-packages (stubs — implementation added week-by-week per DEV_PLAN.md):
  orchestration  — Coordinator, Planner, Reflexion controller
  agents         — Specialist extractors + Critic/Verifier
  parsers        — Docling / Marker / Nougat / PyMuPDF ensemble
  linkers        — ROR, OpenAlex, S2AND author-disambiguation
  kg             — KuzuDB schema + ingestion + GraphRAG queries
  analytics      — Trend analytics, country heatmaps, GNN forecasting
  eval           — PaperAffilBench harness, baselines, metrics
"""

__version__ = "2.0.0-dev"
