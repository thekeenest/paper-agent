"""
Acknowledgements specialist extractor.

Operates exclusively on the acknowledgements section body spans.
The ack section often reveals funding agencies, hosting institutions,
and sometimes author affiliations not listed in the header.
"""
from __future__ import annotations

import structlog

from src.v2.parsers.schemas import ParsedDoc

from ._base import BaseExtractor, _LLMOutput, _span_id

_LOG = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a specialist that extracts author-affiliation information from the
**Acknowledgements section** of academic papers.

## Input
You will receive the text of the acknowledgements section.

## Your task
Extract (author_name, affiliation) pairs that are explicitly named.

## Rules
1. Only extract real person names (not "the authors" or "we").
2. Affiliation comes from named institutions or grants that are tied to
   a specific person in the text.
3. Generic funding acknowledgements ("Supported by NSF grant 123") without
   a named person → skip.
4. "X is supported by [Institution]" → author_name=X, affiliation=[Institution].
5. Do NOT infer names from pronouns or initials alone.
6. Return empty list if no clear named-person affiliations are present.

Respond only with valid JSON matching the output schema.
"""


class AcknowledgementsExtractor(BaseExtractor):
    _SYSTEM_PROMPT = _SYSTEM_PROMPT
    _SPECIALIST = "acknowledgements"

    async def extract(self, doc: ParsedDoc) -> Candidates:  # type: ignore[name-defined]  # noqa: F821
        from src.v2.orchestration.contracts import Candidates
        from langchain_core.messages import HumanMessage, SystemMessage

        spans = doc.acknowledgements
        if not spans:
            _LOG.info("ack_extractor.no_spans")
            return Candidates(items=[], specialist=self._SPECIALIST)

        span_ids = [_span_id(s.text, s.page, self._SPECIALIST) for s in spans]
        numbered = "\n".join(f"[{i+1}] (page {s.page}) {s.text}" for i, s in enumerate(spans))

        user_text = (
            f"Acknowledgements section ({len(spans)} span(s)):\n\n"
            f"{numbered}\n\n"
            "Extract named author-affiliation pairs from the above text only."
        )

        _LOG.info("ack_extractor.start", n_spans=len(spans))

        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_text)]
        raw = await self._chain.ainvoke(messages)
        if not isinstance(raw, _LLMOutput):
            raw = _LLMOutput(candidates=[])

        result = self._to_candidates(raw, span_ids, token_usage={})
        _LOG.info("ack_extractor.done", n_candidates=len(result.items))
        return result
