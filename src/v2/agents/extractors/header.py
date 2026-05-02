"""
Header specialist extractor.

Operates exclusively on the spans the parser ensemble flagged as belonging
to the header region (title + author block at the top of page 0).

Forbidden: inventing any information not present in the input spans.
"""
from __future__ import annotations

import structlog

from src.v2.parsers.schemas import ParsedDoc

from ._base import BaseExtractor, _span_id

_LOG = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a specialist that extracts author names and affiliations from the
**header region** of academic papers.

## Input
You will receive a numbered list of text spans from the title/author block
(typically the first few lines of page 1 of an academic paper).

## Your task
Extract the authors and their direct affiliations from these spans ONLY.
Do NOT infer anything not written in the provided text.

## Rules
1. Each unique author name → one candidate.
2. Affiliations: take the institution strings that are explicitly linked to
   each author via superscript numbers, symbols, or proximity.
3. If an affiliation is shared by multiple authors, repeat it for each.
4. Do NOT make up or complete partial names.
5. Reject initials-only entries (e.g. "A. B." alone is fine, but "A." alone
   is not a full name — skip it).
6. Emails in the header may be assigned to the nearest preceding author name.
7. Return an empty list if no authors are identifiable.

Respond only with valid JSON matching the output schema.
"""


class HeaderExtractor(BaseExtractor):
    """Extract authors/affiliations from header spans only."""

    _SYSTEM_PROMPT = _SYSTEM_PROMPT
    _SPECIALIST = "header"

    async def extract(self, doc: ParsedDoc, retry_hint: str = "") -> Candidates:  # type: ignore[name-defined]  # noqa: F821
        from src.v2.orchestration.contracts import Candidates

        spans = doc.headers
        if not spans:
            _LOG.info("header_extractor.no_spans")
            return Candidates(items=[], specialist=self._SPECIALIST)

        span_ids = [_span_id(s.text, s.page, self._SPECIALIST) for s in spans]
        numbered = "\n".join(f"[{i+1}] (page {s.page}) {s.text}" for i, s in enumerate(spans))

        user_text = (
            f"Header spans from the paper ({len(spans)} total):\n\n"
            f"{numbered}\n\n"
            "Extract authors and affiliations from the above spans only."
        )

        _LOG.info("header_extractor.start", n_spans=len(spans))

        from ._base import _LLMOutput
        messages = self._build_messages(user_text, retry_hint=retry_hint)

        raw = await self._chain.ainvoke(messages)
        if not isinstance(raw, _LLMOutput):
            raw = _LLMOutput(candidates=[])

        result = self._to_candidates(raw, span_ids, token_usage={})
        _LOG.info("header_extractor.done", n_candidates=len(result.items))
        return result
