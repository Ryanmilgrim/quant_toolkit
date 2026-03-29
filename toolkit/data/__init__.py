"""Data access helpers."""

from .fred import FredConfig, fetch_fred_series, get_fred_series_info, search_fred_series
from .french_industry import (
    SUPPORTED_FACTOR_SETS,
    SUPPORTED_INDUSTRY_UNIVERSES,
    FactorSet,
    fetch_ff_factors_daily,
    fetch_ff_industry_daily,
)

__all__ = [
    "FredConfig",
    "SUPPORTED_FACTOR_SETS",
    "SUPPORTED_INDUSTRY_UNIVERSES",
    "FactorSet",
    "fetch_ff_factors_daily",
    "fetch_ff_industry_daily",
    "fetch_fred_series",
    "get_fred_series_info",
    "search_fred_series",
]
