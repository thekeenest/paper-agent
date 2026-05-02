"""
Specialist extractor agents.

Public surface
--------------
::

    from src.v2.agents.extractors import (
        HeaderExtractor,
        FootnoteExtractor,
        AcknowledgementsExtractor,
        EmailDomainExtractor,
        merge_candidates,
    )

Each extractor:
  * receives only its region's spans from ParsedDoc
  * returns Candidates(items: list[Candidate])
  * is forbidden from inventing information not in its input spans

merge_candidates() combines all four outputs, promotes confidence="high"
when ≥2 specialists agree, and builds a full EvidenceTrail.

See DEV_PLAN.md §3.3 and docs/architecture.md.
"""

from .acknowledgements import AcknowledgementsExtractor
from .email_domain import EmailDomainExtractor
from .footnote import FootnoteExtractor
from .header import HeaderExtractor
from .merge import merge_candidates

__all__ = [
    "HeaderExtractor",
    "FootnoteExtractor",
    "AcknowledgementsExtractor",
    "EmailDomainExtractor",
    "merge_candidates",
]
