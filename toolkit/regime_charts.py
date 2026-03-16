"""Matplotlib chart functions for regime detection results.

All plotting functions follow the matplotlib ``ax`` pattern: pass an existing
``Axes`` to draw on, or omit to create a new figure automatically.  Every
function returns the ``Axes`` object so callers can further customise it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolkit.charts import _get_ax

_REGIME_PALETTE = ["#2c9bac", "#e63946", "#457b9d", "#f4a261", "#264653"]


# ---------------------------------------------------------------------------
# Stacked regime probabilities
# ---------------------------------------------------------------------------

def plot_regime_probabilities(
    probabilities: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Smoothed Regime Probabilities",
    colors: Optional[Sequence[str]] = None,
) -> plt.Axes:
    """Stacked area chart of regime probabilities over time.

    Parameters
    ----------
    probabilities
        DataFrame with columns ``regime_0``, ``regime_1``, … and a
        DatetimeIndex.  Values should be in [0, 1].
    """
    ax = _get_ax(ax)
    if probabilities.empty:
        ax.set_title(title)
        return ax

    colors = colors or _REGIME_PALETTE[: probabilities.shape[1]]
    labels = list(probabilities.columns)

    ax.stackplot(
        probabilities.index,
        *[probabilities[col].values for col in probabilities.columns],
        labels=labels,
        colors=colors,
        alpha=0.85,
    )
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, color="#d8dadb", alpha=0.4, linewidth=0.5)

    return ax


# ---------------------------------------------------------------------------
# Series with regime-coloured background
# ---------------------------------------------------------------------------

def plot_regime_series(
    series: pd.Series,
    regime_assignments: pd.Series,
    *,
    k_regimes: int = 2,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
) -> plt.Axes:
    """Plot the original series with background bands coloured by regime.

    Uses ``ax.axvspan`` to draw semi-transparent rectangles behind the data.
    """
    ax = _get_ax(ax)
    if series.empty:
        ax.set_title(title or "Series")
        return ax

    colors = colors or _REGIME_PALETTE[:k_regimes]
    title = title or (series.name or "Series")

    # Draw regime background bands
    regimes = regime_assignments.reindex(series.index).fillna(-1).astype(int)
    prev_regime = int(regimes.iloc[0])
    band_start = series.index[0]

    for i in range(1, len(regimes)):
        curr = int(regimes.iloc[i])
        if curr != prev_regime or i == len(regimes) - 1:
            end_idx = series.index[i] if i < len(regimes) - 1 else series.index[-1]
            if 0 <= prev_regime < len(colors):
                ax.axvspan(
                    band_start, end_idx,
                    color=colors[prev_regime], alpha=0.15,
                )
            band_start = series.index[i]
            prev_regime = curr

    # Plot the series itself
    ax.plot(series.index, series.values, color="#001f38", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(series.name or "Value")
    ax.grid(True, color="#d8dadb", alpha=0.4, linewidth=0.5)

    return ax


# ---------------------------------------------------------------------------
# Transition matrix heatmap
# ---------------------------------------------------------------------------

def plot_transition_matrix(
    matrix: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Transition Matrix",
    cmap: str = "Blues",
) -> plt.Axes:
    """Heatmap of the regime transition probability matrix."""
    ax = _get_ax(ax)

    A = matrix.to_numpy(dtype=float)
    k = A.shape[0]

    im = ax.imshow(A, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # Annotate cells
    for i in range(k):
        for j in range(k):
            val = A[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=text_color)

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(matrix.columns, fontsize=9)
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(title)

    return ax


__all__ = [
    "plot_regime_probabilities",
    "plot_regime_series",
    "plot_transition_matrix",
]
