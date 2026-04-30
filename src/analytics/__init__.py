# Re-export shim for v1 AnalyticsEngine — keeps `from src.analytics import AnalyticsEngine` working.
# New v2 analytics modules (graph_analytics, trend_analytics, …) live as siblings in this package.
# See src/v2/analytics/ for the v2 graph-query analytics layer.
from src.v1.analytics import *  # noqa: F401, F403
from src.v1.analytics import AnalyticsEngine  # noqa: F401
