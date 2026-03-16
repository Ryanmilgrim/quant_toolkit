"""Time series transformations for regime detection inputs.

Separates the data transformation step from the model fitting step,
allowing users to compose any FRED series with any transform before
feeding into a Markov Switching model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class TransformType(Enum):
    """Supported time series transformations."""

    NONE = "none"
    FIRST_DIFF = "first_diff"
    PCT_CHANGE = "pct_change"
    LOG_DIFF = "log_diff"
    YOY_CHANGE = "yoy_change"
    ROLLING_MEAN = "rolling_mean"


TRANSFORM_DESCRIPTIONS: dict[TransformType, str] = {
    TransformType.NONE: "No transformation (raw levels)",
    TransformType.FIRST_DIFF: "First difference (x_t - x_{t-1})",
    TransformType.PCT_CHANGE: "Percentage change ((x_t - x_{t-1}) / x_{t-1})",
    TransformType.LOG_DIFF: "Log difference (ln(x_t) - ln(x_{t-1}))",
    TransformType.YOY_CHANGE: "Year-over-year change (x_t - x_{t-window})",
    TransformType.ROLLING_MEAN: "Rolling mean over window periods",
}


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for a time series transformation.

    Parameters
    ----------
    transform
        The type of transformation to apply.
    window
        Window size for rolling-mean or year-over-year transforms.
        For YoY with quarterly data use 4, with monthly data use 12.
    """

    transform: TransformType = TransformType.NONE
    window: int = 4

    @property
    def description(self) -> str:
        base = TRANSFORM_DESCRIPTIONS.get(self.transform, str(self.transform))
        if self.transform in (TransformType.YOY_CHANGE, TransformType.ROLLING_MEAN):
            return f"{base} (window={self.window})"
        return base


def apply_transform(series: pd.Series, config: TransformConfig) -> pd.Series:
    """Apply a transformation to a time series.

    Parameters
    ----------
    series
        Input time series (typically raw FRED data).
    config
        Transformation configuration.

    Returns
    -------
    pd.Series
        Transformed series with NaN values dropped.
    """
    t = config.transform

    if t == TransformType.NONE:
        result = series.copy()
    elif t == TransformType.FIRST_DIFF:
        result = series.diff()
    elif t == TransformType.PCT_CHANGE:
        result = series.pct_change()
    elif t == TransformType.LOG_DIFF:
        result = np.log(series).diff()
    elif t == TransformType.YOY_CHANGE:
        result = series.diff(config.window)
    elif t == TransformType.ROLLING_MEAN:
        result = series.rolling(config.window).mean()
    else:
        raise ValueError(f"Unknown transform type: {t}")

    return result.dropna()


__all__ = [
    "TRANSFORM_DESCRIPTIONS",
    "TransformConfig",
    "TransformType",
    "apply_transform",
]
