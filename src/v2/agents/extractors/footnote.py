"""
Footnote specialist extractor.

Operates exclusively on the footnote spans (small-font text at page bottoms
that often contain affiliation superscript expansions and correspondence notes).
"""
from __future__ import annotations

import structlog

from src.v2.parsers.schemas import ParsedDoc

from ._base import BaseExtractor, _LLMOutput, _span_id

_LOG = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a specialist that extracts author-affiliation information from the
**footnote region** of academic papers.

## Input
You will receive footnote text spans from a PDF — these typically contain:
  - Superscript-numbered affiliation expansions (e.g. "¹ MIT CSAIL, Cambridge")
  - Correspondence notes ("Corresponding author: alice@mit.edu")
  - Present-address notes ("² Now at OpenAI, San Francisco")

## Your task
Extract (author_name, affiliation) pairs from these footnotes ONLY.

## Rules
1. Affiliation-only footnotes (no name) → affiliation goes into the candidate
   whose superscript number matches.  If no name is derivable, skip it.
2. Correspondence notes → assign the email to the named author.
3. "Now at" / "Work done at" → treat as an additional affiliation for that author.
4. Do NOT infer author names from email address prefixes.
5. Return empty list if no clear author-affiliation pairs are present.

Respond only with valid JSON matching the output schema.
"""


class FootnoteExtractor(BaseExtractor):
    _SYSTEM_PROMPT = _SYSTEM_PROMPT
    _SPECIALIST = "footnote"

    async def extract(self, doc: ParsedDoc) -> Candidates:  # type: ignore[name-defined]  # noqa: F821
        from src.v2.orchestration.contracts import Candidates
        from langchain_core.messages import HumanMessage, SystemMessage

        spans = doc.footnotes
        if not spans:
            _LOG.info("footnote_extractor.no_spans")
            return Candidates(items=[], specialist=self._SPECIALIST)

        span_ids = [_span_id(s.text, s.page, self._SPECIALIST) for s in spans]
        numbered = "\n".join(f"[{i+1}] (page {s.page}) {s.text}" for i, s in enumerate(spans))

        user_text = (
            f"Footnote spans ({len(spans)} total):\n\n"
            f"{numbered}\n\n"
            "Extract author-affiliation pairs from the above footnotes only."
        )

        _LOG.info("footnote_extractor.start", n_spans=len(spans))

        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_text)]
        raw = await self._chain.ainvoke(messages)
        if not isinstance(raw, _LLMOutput):
            raw = _LLMOutput(candidates=[])

        result = self._to_candidates(raw, span_ids, token_usage={})
        _LOG.info("footnote_extractor.done", n_candidates=len(result.items))
        return result
