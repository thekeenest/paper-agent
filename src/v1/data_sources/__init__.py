"""
Data Sources Module - Интеграция с различными источниками научных публикаций.

Поддерживаемые источники:
- ArXiv (по умолчанию)
- Semantic Scholar
- OpenAlex
- ROR (Research Organization Registry)
"""

from .base import DataSourceBase, DataSourceType, SearchParams
from .arxiv_client import ArxivClient
from .semantic_scholar import SemanticScholarClient
from .openalex import OpenAlexClient
from .ror import RORLookup, get_ror_lookup, lookup_ror
from .router import DataSourceRouter, get_data_router

__all__ = [
    "DataSourceBase",
    "DataSourceType",
    "SearchParams",
    "ArxivClient",
    "SemanticScholarClient",
    "OpenAlexClient",
    "RORLookup",
    "get_ror_lookup",
    "lookup_ror",
    "DataSourceRouter",
    "get_data_router",
]
