import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolkit.risk_charts import (
    plot_agg_correlation_backtest,
    plot_asset_residuals_and_vol,
    plot_beta_heatmap,
    plot_factor_risk_heatmap,
    plot_factor_volatility_backtest,
    plot_returns_with_confidence_bands,
    plot_volatility_regression_scatter,
)


def _dates(n: int = 100):
    return pd.bdate_range("2020-01-01", periods=n)


def _series(n: int = 100, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.01, n), index=_dates(n))


def _dataframe(n: int = 100, cols: int = 5, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, (n, cols)),
        index=_dates(n),
        columns=[f"A{i}" for i in range(cols)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plot_beta_heatmap_returns_axes() -> None:
    betas = pd.DataFrame(
        np.random.default_rng(0).normal(size=(10, 5)),
        index=[f"A{i}" for i in range(10)],
        columns=[f"F{i}" for i in range(5)],
    )
    ax = plot_beta_heatmap(betas)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_beta_heatmap_with_existing_ax() -> None:
    betas = pd.DataFrame(
        np.random.default_rng(1).normal(size=(5, 3)),
        index=[f"A{i}" for i in range(5)],
        columns=[f"F{i}" for i in range(3)],
    )
    fig, ax = plt.subplots()
    result = plot_beta_heatmap(betas, ax=ax)
    assert result is ax
    plt.close("all")


def test_plot_factor_risk_heatmap_correlation() -> None:
    corr = pd.DataFrame(
        np.eye(3), index=["F1", "F2", "F3"], columns=["F1", "F2", "F3"],
    )
    ax = plot_factor_risk_heatmap(corr, metric="correlation")
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_factor_risk_heatmap_covariance() -> None:
    cov = pd.DataFrame(
        np.diag([0.01, 0.02, 0.03]),
        index=["F1", "F2", "F3"], columns=["F1", "F2", "F3"],
    )
    ax = plot_factor_risk_heatmap(cov, metric="covariance")
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_factor_volatility_backtest_returns_axes() -> None:
    pred = _series(100, seed=10).abs()
    real = _series(100, seed=11).abs()
    ax = plot_factor_volatility_backtest(pred, real)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_factor_volatility_backtest_no_annualize() -> None:
    pred = _series(50, seed=20).abs()
    real = _series(50, seed=21).abs()
    ax = plot_factor_volatility_backtest(pred, real, annualize=False)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_returns_with_confidence_bands() -> None:
    r = _series(100, seed=30)
    vol = _series(100, seed=31).abs() + 0.001
    ax = plot_returns_with_confidence_bands(r, vol)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_returns_with_confidence_bands_empty() -> None:
    r = pd.Series(dtype=float)
    vol = pd.Series(dtype=float)
    ax = plot_returns_with_confidence_bands(r, vol)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_volatility_regression_scatter() -> None:
    pred = _dataframe(100, 5, seed=40).abs()
    real = _dataframe(100, 5, seed=41).abs()
    ax = plot_volatility_regression_scatter(pred, real, oos_only=False)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_agg_correlation_backtest() -> None:
    pred = _series(80, seed=50)
    real = _series(80, seed=51)
    ax = plot_agg_correlation_backtest(pred, real, asset="test")
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_asset_residuals_and_vol_single_ax() -> None:
    std_resid = _series(100, seed=60)
    vol = _series(100, seed=61).abs() + 0.001
    fig, ax = plt.subplots()
    result = plot_asset_residuals_and_vol(std_resid, vol, asset="A0", ax=ax)
    assert result is ax
    plt.close("all")


def test_plot_asset_residuals_and_vol_two_panel() -> None:
    std_resid = _series(100, seed=70)
    vol = _series(100, seed=71).abs() + 0.001
    ax = plot_asset_residuals_and_vol(std_resid, vol, asset="A0")
    assert isinstance(ax, plt.Axes)
    plt.close("all")
