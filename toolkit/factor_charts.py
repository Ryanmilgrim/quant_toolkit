"""Matplotlib chart functions for factor analysis results.

All plotting functions follow the matplotlib ``ax`` pattern: pass an existing
``Axes`` to draw on, or omit to create a new figure automatically.  Every
function returns the ``Axes`` object so callers can further customise it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolkit.charts import _get_ax

_PALETTE = {
    "forecast": "#e63946",
    "realized": "#457b9d",
    "return": "#001f38",
    "confidence": "#a8dadc",
    "identity": "#6c757d",
    "regression": "#e63946",
    "grid": "#d8dadb",
}


def _annualize_vol(
    vol: Union[pd.Series, pd.DataFrame],
    *,
    trading_days: int = 252,
) -> Union[pd.Series, pd.DataFrame]:
    """Annualize a daily volatility series."""
    return vol * float(np.sqrt(int(trading_days)))


# ---------------------------------------------------------------------------
# Beta heatmap
# ---------------------------------------------------------------------------

def plot_beta_heatmap(
    betas: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    max_assets: int = 60,
    max_factors: int = 30,
    title: str = "Beta loadings heatmap",
) -> plt.Axes:
    """Heatmap of factor beta loadings."""
    ax = _get_ax(ax)

    betas = betas.iloc[:int(max_assets), :int(max_factors)]
    A = betas.to_numpy(dtype=float)

    vmax = float(np.nanpercentile(np.abs(A), 98)) if np.isfinite(A).any() else 1.0
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0

    im = ax.imshow(A, aspect="auto", interpolation="nearest",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, label="Beta")
    ax.set_yticks(range(len(betas.index)))
    ax.set_yticklabels(betas.index, fontsize=7)
    ax.set_xticks(range(len(betas.columns)))
    ax.set_xticklabels(betas.columns, rotation=90, fontsize=7)
    ax.set_title(title)

    return ax


# ---------------------------------------------------------------------------
# Factor risk heatmap (correlation or covariance)
# ---------------------------------------------------------------------------

def plot_factor_risk_heatmap(
    matrix: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    metric: str = "correlation",
    title: Optional[str] = None,
) -> plt.Axes:
    """Heatmap of a factor risk matrix (correlation or covariance).

    Parameters
    ----------
    matrix
        Pre-computed correlation or covariance DataFrame (square, labelled).
    metric
        ``"correlation"`` or ``"covariance"`` — controls colour-bar range.
    """
    ax = _get_ax(ax)

    M = matrix.to_numpy(dtype=float)
    if metric == "correlation":
        vmin, vmax = -1.0, 1.0
        cbar_label = "Correlation"
    else:
        vmax = float(np.nanpercentile(np.abs(M), 98)) if np.isfinite(M).any() else 1.0
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        vmin = -vmax
        cbar_label = "Covariance"

    if title is None:
        title = f"Factor {metric} heatmap"

    im = ax.imshow(M, aspect="auto", interpolation="nearest",
                   cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title(title)

    return ax


# ---------------------------------------------------------------------------
# Volatility backtest (predicted vs trailing)
# ---------------------------------------------------------------------------

def plot_factor_volatility_backtest(
    predicted_vol: pd.Series,
    realized_vol: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    train_end: Optional[Union[pd.Timestamp, str]] = None,
    title: str = "Predicted vs trailing volatility",
    label_pred: str = "predicted",
    label_real: str = "realized",
    annualize: bool = True,
    trading_days: int = 252,
    y_label: Optional[str] = None,
) -> plt.Axes:
    """Line chart comparing predicted conditional vol with a trailing benchmark."""
    ax = _get_ax(ax)

    pred = predicted_vol.astype(float)
    real = realized_vol.astype(float)

    if annualize:
        pred = _annualize_vol(pred, trading_days=trading_days)
        real = _annualize_vol(real, trading_days=trading_days)

    idx = pred.index.intersection(real.index)
    pred = pred.loc[idx]
    real = real.loc[idx]

    ax.plot(pred.index, pred.to_numpy(dtype=float), label=label_pred,
            linewidth=1.2, alpha=0.9, color=_PALETTE["forecast"])
    ax.plot(real.index, real.to_numpy(dtype=float), label=label_real,
            linewidth=1.2, alpha=0.9, color=_PALETTE["realized"])

    if train_end is not None:
        ax.axvline(pd.Timestamp(train_end), linewidth=1.0, alpha=0.7,
                   color=_PALETTE["identity"])

    ax.set_title(title)
    if y_label is None:
        y_label = "vol (ann.)" if annualize else "vol (daily)"
    ax.set_ylabel(str(y_label))
    ax.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)
    ax.legend(loc="best")

    return ax


# ---------------------------------------------------------------------------
# Returns with confidence bands
# ---------------------------------------------------------------------------

def plot_returns_with_confidence_bands(
    returns: pd.Series,
    cond_vol: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    n_std: Sequence[float] = (2.0, 3.0),
    train_end: Optional[Union[pd.Timestamp, str]] = None,
    oos_only: bool = False,
    title: str = "Daily returns with conditional volatility bands",
) -> plt.Axes:
    """Plot daily returns with +/- k*sigma conditional volatility bands."""
    ax = _get_ax(ax)

    r = returns.astype(float)
    s = cond_vol.astype(float).clip(lower=0.0)

    idx = r.index.intersection(s.index)
    if train_end is not None and oos_only:
        idx = idx[idx >= pd.Timestamp(train_end)]
    r = r.loc[idx]
    s = s.loc[idx]

    if len(idx) == 0:
        ax.set_title(title)
        return ax

    bands = sorted(float(x) for x in n_std)

    ax.plot(r.index, r.to_numpy(dtype=float), linewidth=0.9, alpha=0.85,
            label="daily return", color=_PALETTE["return"])
    ax.axhline(0.0, linewidth=1.0, alpha=0.6, color=_PALETTE["identity"])

    for k in bands:
        upper = k * s
        lower = -k * s
        ax.plot(upper.index, upper.to_numpy(dtype=float), linewidth=0.9,
                alpha=0.7, label=f"+{k:g}\u03c3", color=_PALETTE["confidence"])
        ax.plot(lower.index, lower.to_numpy(dtype=float), linewidth=0.9,
                alpha=0.7, label=f"-{k:g}\u03c3", color=_PALETTE["confidence"])

    if train_end is not None:
        ax.axvline(pd.Timestamp(train_end), linewidth=1.0, alpha=0.7,
                   color=_PALETTE["identity"])

    ax.set_title(title)
    ax.set_ylabel("daily return")
    ax.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)
    ax.legend(loc="best", ncols=3, fontsize="small")

    return ax


# ---------------------------------------------------------------------------
# Volatility regression scatter
# ---------------------------------------------------------------------------

def plot_volatility_regression_scatter(
    pred_vol: pd.DataFrame,
    real_vol: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    train_end: Optional[Union[pd.Timestamp, str]] = None,
    annualize: bool = True,
    trading_days: int = 252,
    oos_only: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """Scatter + regression: predicted vs trailing vol (one point per asset)."""
    ax = _get_ax(ax)

    idx = pred_vol.index.intersection(real_vol.index)
    if oos_only and train_end is not None:
        idx = idx[idx >= pd.Timestamp(train_end)]
    pv = pred_vol.loc[idx]
    rv = real_vol.loc[idx]

    if annualize:
        pv = _annualize_vol(pv, trading_days=trading_days)
        rv = _annualize_vol(rv, trading_days=trading_days)

    x = rv.mean(axis=0)
    y = pv.mean(axis=0)
    df = pd.concat([x.rename("real"), y.rename("pred")], axis=1).dropna()
    if df.empty:
        ax.set_title(title or "Volatility scatter")
        return ax

    xv = df["real"].to_numpy(dtype=float)
    yv = df["pred"].to_numpy(dtype=float)

    b, a = np.polyfit(xv, yv, deg=1)
    y_hat = a + b * xv
    ss_res = float(np.sum((yv - y_hat) ** 2))
    ss_tot = float(np.sum((yv - float(np.mean(yv))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    lo = float(np.nanmin(np.r_[xv, yv]))
    hi = float(np.nanmax(np.r_[xv, yv]))

    ax.scatter(xv, yv, alpha=0.85, color=_PALETTE["realized"])
    ax.plot([lo, hi], [lo, hi], linewidth=1.0, alpha=0.6,
            color=_PALETTE["identity"], label="y = x")
    ax.plot([lo, hi], [a + b * lo, a + b * hi], linewidth=1.2, alpha=0.8,
            color=_PALETTE["regression"],
            label=f"fit: y = {a:.3g} + {b:.3g}x")

    for name, (xx, yy) in df.iterrows():
        ax.text(float(xx), float(yy), str(name), fontsize=7, alpha=0.85)

    ann = " (ann.)" if annualize else " (daily)"
    ax.set_xlabel(f"Trailing volatility{ann}")
    ax.set_ylabel(f"Predicted conditional volatility{ann}")

    if title is None:
        title = f"Predicted vs trailing vol (1 point per asset) | R\u00b2={r2:.3f}"
    ax.set_title(title)
    ax.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)
    ax.legend(loc="best")

    return ax


# ---------------------------------------------------------------------------
# Aggregate correlation backtest
# ---------------------------------------------------------------------------

def plot_agg_correlation_backtest(
    pred_corr: pd.Series,
    real_corr: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    asset: str = "",
    train_end: Optional[Union[pd.Timestamp, str]] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Backtest corr(equal-weighted aggregate, asset)."""
    ax = _get_ax(ax)

    pred = pred_corr.astype(float)
    real = real_corr.astype(float)
    idx = pred.index.intersection(real.index)
    pred = pred.loc[idx]
    real = real.loc[idx]

    ax.plot(pred.index, pred.to_numpy(dtype=float),
            label="pred corr(agg, asset)", linewidth=1.2, alpha=0.9,
            color=_PALETTE["forecast"])
    ax.plot(real.index, real.to_numpy(dtype=float),
            label="trailing corr(agg, asset)", linewidth=1.2, alpha=0.9,
            color=_PALETTE["realized"])

    if train_end is not None:
        ax.axvline(pd.Timestamp(train_end), linewidth=1.0, alpha=0.7,
                   color=_PALETTE["identity"])

    if title is None:
        title = f"corr(agg, {asset}) predicted vs trailing"
    ax.set_title(title)
    ax.set_ylabel("correlation")
    ax.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)
    ax.legend(loc="best")

    return ax


# ---------------------------------------------------------------------------
# Asset residuals and conditional vol
# ---------------------------------------------------------------------------

def plot_asset_residuals_and_vol(
    std_resid: pd.Series,
    cond_vol: pd.Series,
    *,
    asset: str = "",
    annualize: bool = True,
    trading_days: int = 252,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Standardized residuals and conditional vol for a single asset.

    Draws two subplots on the same figure: standardized innovations
    (top) and conditional volatility (bottom).  If *ax* is provided the
    standardized innovations are drawn there and vol is skipped.
    """
    if ax is not None:
        # Single-axes mode: draw std_resid only
        ax.plot(std_resid.index, std_resid.to_numpy(dtype=float),
                linewidth=0.9, alpha=0.85, color=_PALETTE["return"])
        ax.axhline(0.0, linewidth=1.0, alpha=0.6, color=_PALETTE["identity"])
        ax.set_title(title or f"Standardized residual innovations: {asset}")
        ax.set_ylabel("z")
        ax.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)
        return ax

    # Two-panel mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(std_resid.index, std_resid.to_numpy(dtype=float),
             linewidth=0.9, alpha=0.85, color=_PALETTE["return"])
    ax1.axhline(0.0, linewidth=1.0, alpha=0.6, color=_PALETTE["identity"])
    ax1.set_title(title or f"Standardized residual innovations: {asset}")
    ax1.set_ylabel("z")
    ax1.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)

    vol = cond_vol.astype(float)
    if annualize:
        vol = _annualize_vol(vol, trading_days=trading_days)
    ax2.plot(vol.index, vol.to_numpy(dtype=float),
             linewidth=1.0, alpha=0.85, color=_PALETTE["forecast"])
    ax2.set_title(f"Residual conditional volatility: {asset}")
    ax2.set_ylabel("sigma (ann.)" if annualize else "sigma (daily)")
    ax2.grid(True, color=_PALETTE["grid"], alpha=0.4, linewidth=0.5)

    fig.tight_layout()
    return ax1


__all__ = [
    "plot_agg_correlation_backtest",
    "plot_asset_residuals_and_vol",
    "plot_beta_heatmap",
    "plot_factor_risk_heatmap",
    "plot_factor_volatility_backtest",
    "plot_returns_with_confidence_bands",
    "plot_volatility_regression_scatter",
]
