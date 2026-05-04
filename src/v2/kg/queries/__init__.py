"""
KG query modules for Paper-Agent v2.

Each module exposes a single function returning a typed result.

Available queries
-----------------
  industry_share_by_venue   — fraction of industry-affiliated authors per venue/year
  country_strategy_heatmap  — collaboration counts by country-pair per year
  coauthor_neighborhood     — 1-/2-hop co-author subgraph for a given author
  top_institutions_by_venue — ranked institutions by paper count for a venue/year
"""
from .coauthor_neighborhood import coauthor_neighborhood, CoauthorNeighborhoodResult
from .country_strategy_heatmap import country_strategy_heatmap, CountryHeatmapResult
from .industry_share_by_venue import industry_share_by_venue, IndustryShareResult
from .top_institutions_by_venue import top_institutions_by_venue, TopInstitutionsResult

__all__ = [
    "coauthor_neighborhood",
    "CoauthorNeighborhoodResult",
    "country_strategy_heatmap",
    "CountryHeatmapResult",
    "industry_share_by_venue",
    "IndustryShareResult",
    "top_institutions_by_venue",
    "TopInstitutionsResult",
]
