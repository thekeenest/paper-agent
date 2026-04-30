# src/v2/agents/critic/

**Purpose:** Grounded Critic/Verifier — the methodological core of v2.

The Critic must *cite evidence* (OpenAlex, ROR, S2, DNS, page-text) for every accept/reject.
An undocumented decision is treated as a soft-reject and triggers a Reflexion update.

Planned modules:
- `critic.py` — main verifier LangGraph node
- `tools.py` — retrieval tools: OpenAlex lookup, ROR search, DNS resolver, S2 author endpoint
- `schemas.py` — `EvidenceRecord`, `CriticVerdict`, `VerificationReport`

See [DEV_PLAN.md §3.4](../../../../coursework_v2/DEV_PLAN.md) and
[READING_LIST.md §A (CRITIC, MAR, CGI)](../../../../coursework_v2/READING_LIST.md).
