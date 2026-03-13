"""Matplotlib chart functions for quantitative analysis results.

All plotting functions follow the matplotlib ``ax`` pattern: pass an existing
``Axes`` to draw on, or omit to create a new figure automatically.  Every
function returns the ``Axes`` object so callers can further customise it.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PALETTE = {
    "portfolio": "#001f38",
    "benchmark": "#2c9bac",
    "control": "#ff7a00",
    "grid": "#d8dadb",
}

STEPS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}


def _get_ax(ax: Optional[plt.Axes]) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def performance_summary(
    returns: pd.Series,
    *,
    steps_per_year: int = 252,
) -> dict[str, Optional[float]]:
    """Compute key performance statistics from a return series.

    Returns a dict with total_return, ann_return, ann_vol, and sharpe
    (all expressed as percentages except sharpe which is a ratio).
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return {
            "total_return": None,
            "ann_return": None,
            "ann_vol": None,
            "sharpe": None,
        }

    total_return = float(r.sum() * 100.0)
    ann_return = float(r.mean() * steps_per_year * 100.0)

    vol = float(r.std())
    ann_vol = float(vol * np.sqrt(steps_per_year) * 100.0) if np.isfinite(vol) else None

    if np.isfinite(vol) and vol > 0:
        sharpe = float((r.mean() / vol) * np.sqrt(steps_per_year))
    else:
        sharpe = None

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }


def plot_growth(
    portfolio_return: pd.Series,
    benchmark_return: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Cumulative Growth (base = 100)",
) -> plt.Axes:
    """Plot indexed cumulative growth for portfolio vs benchmark."""
    ax = _get_ax(ax)

    combined = pd.concat(
        [portfolio_return.rename("Portfolio"), benchmark_return.rename("Benchmark")],
        axis=1,
    ).dropna(how="any")

    if combined.empty:
        ax.set_title(title)
        return ax

    growth = np.exp(combined.cumsum()) * 100

    ax.plot(growth.index, growth["Portfolio"], color=_PALETTE["portfolio"], linewidth=1.2, label="Portfolio")
    ax.plot(growth.index, growth["Benchmark"], color=_PALETTE["benchmark"], linewidth=1.2, label="Benchmark")
    ax.set_title(title)
    ax.set_ylabel("Indexed growth")
    ax.legend()
    ax.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)

    return ax


def plot_tracking_error(
    tracking_error: pd.Series,
    *,
    control_sd: float = 3.0,
    ax: Optional[plt.Axes] = None,
    title: str = "Residual Return",
) -> plt.Axes:
    """Plot residual return with control limits."""
    ax = _get_ax(ax)

    te = tracking_error.dropna() * 100.0
    if te.empty:
        ax.set_title(title)
        return ax

    ax.plot(te.index, te.values, color=_PALETTE["portfolio"], linewidth=0.8, label="Residual return")

    sd = float(te.std())
    if np.isfinite(sd) and sd > 0:
        upper = control_sd * sd
        lower = -control_sd * sd
        ax.axhline(upper, color=_PALETTE["control"], linestyle="--", linewidth=0.8, label=f"+{control_sd:.0f}\u03c3")
        ax.axhline(lower, color=_PALETTE["control"], linestyle="--", linewidth=0.8, label=f"-{control_sd:.0f}\u03c3")

    ax.set_title(title)
    ax.set_ylabel("Daily return (%)")
    ax.legend()
    ax.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)

    return ax


def _top_weights(weights: pd.Series, *, top_n: int = 10) -> pd.Series:
    """Normalise and truncate a weight vector, grouping tail into 'Other'."""
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    w = weights.astype(float).clip(lower=0.0)
    s = float(w.sum())
    if s <= 0:
        return pd.Series(dtype=float)
    w = w / s
    ranked = w.sort_values(ascending=False)
    top = ranked.head(top_n).copy()
    other = float(ranked.iloc[top_n:].sum())
    if other > 0:
        top.loc["Other"] = other
    return top[top > 0]


def plot_weights_history(
    weights: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    top_n: int = 10,
    title: str = "Historical Style Weights",
) -> plt.Axes:
    """Stacked area chart of rolling style weights over time."""
    ax = _get_ax(ax)

    if weights.empty:
        ax.set_title(title)
        return ax

    data = weights.copy().astype(float).clip(lower=0.0)
    row_sum = data.sum(axis=1).replace(0.0, np.nan)
    data = data.div(row_sum, axis=0).fillna(0.0) * 100.0

    ax.stackplot(data.index, *[data[col].values for col in data.columns], labels=data.columns, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("Weight (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    ax.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)

    return ax


def plot_weights_snapshot(
    weights_series: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    top_n: int = 10,
    title: str = "Current Weight Allocation",
) -> plt.Axes:
    """Horizontal bar chart of the latest weight allocation."""
    ax = _get_ax(ax)

    top = _top_weights(weights_series, top_n=top_n)
    if top.empty:
        ax.set_title(title)
        return ax

    top_sorted = top.sort_values(ascending=True)
    colors = [_PALETTE["portfolio"]] * len(top_sorted)

    ax.barh(top_sorted.index, top_sorted.values * 100, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Weight (%)")
    ax.grid(True, axis="x", color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)

    return ax


__all__ = [
    "STEPS_PER_YEAR",
    "performance_summary",
    "plot_growth",
    "plot_tracking_error",
    "plot_weights_history",
    "plot_weights_snapshot",
]
