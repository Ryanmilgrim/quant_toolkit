"""Quantitative finance toolkit — pure Python, no web framework dependencies."""

from toolkit.analysis import (
    FactorModel,
    FactorRun,
    METHOD_PROJECTION,
    METHOD_QP,
    StyleAnalysis,
    StyleAnalysisSnapshot,
    StyleRun,
    StyleSnapshotInfo,
    black_scholes_price,
    list_style_snapshots,
    load_style_snapshot,
    normalize_snapshot_name,
    save_style_snapshot,
    snapshot_path,
)
from toolkit.data import (
    SUPPORTED_FACTOR_SETS,
    SUPPORTED_INDUSTRY_UNIVERSES,
    FactorSet,
    fetch_ff_factors_daily,
    fetch_ff_industry_daily,
)
from toolkit.returns import to_log_returns
from toolkit.universe import get_universe_returns, get_universe_start_date

__all__ = [
    "FactorModel",
    "FactorRun",
    "METHOD_PROJECTION",
    "METHOD_QP",
    "StyleAnalysis",
    "StyleAnalysisSnapshot",
    "StyleRun",
    "StyleSnapshotInfo",
    "SUPPORTED_FACTOR_SETS",
    "SUPPORTED_INDUSTRY_UNIVERSES",
    "FactorSet",
    "black_scholes_price",
    "fetch_ff_factors_daily",
    "fetch_ff_industry_daily",
    "get_universe_returns",
    "get_universe_start_date",
    "list_style_snapshots",
    "load_style_snapshot",
    "normalize_snapshot_name",
    "save_style_snapshot",
    "snapshot_path",
    "to_log_returns",
]
