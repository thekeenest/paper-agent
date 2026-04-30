# Re-export shim — delegates to the frozen v1 data-source implementations.
# Do not add logic here; v2 source clients live in src/v2/linkers/ (ROR, OpenAlex) and
# src/v2/parsers/ (Docling, Marker, Nougat).
from src.v1.data_sources import *  # noqa: F401, F403
from src.v1.data_sources import (  # noqa: F401
    DataSourceBase,
    DataSourceType,
    SearchParams,
    ArxivClient,
    SemanticScholarClient,
    OpenAlexClient,
    RORLookup,
    get_ror_lookup,
    lookup_ror,
    DataSourceRouter,
    get_data_router,
)
