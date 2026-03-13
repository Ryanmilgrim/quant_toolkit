import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from quant_toolkit.charts import (
    performance_summary,
    plot_growth,
    plot_tracking_error,
    plot_weights_history,
    plot_weights_snapshot,
)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    port = pd.Series(rng.normal(0.0005, 0.01, 100), index=dates, name="portfolio")
    bench = pd.Series(rng.normal(0.0004, 0.01, 100), index=dates, name="benchmark")
    return port, bench


@pytest.fixture
def sample_weights():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    rng = np.random.default_rng(42)
    w = rng.dirichlet(np.ones(5), size=12)
    return pd.DataFrame(w, index=dates, columns=[f"A{i}" for i in range(5)])


def test_plot_growth_returns_axes(sample_returns):
    port, bench = sample_returns
    ax = plot_growth(port, bench)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_growth_with_existing_ax(sample_returns):
    port, bench = sample_returns
    fig, ax = plt.subplots()
    result = plot_growth(port, bench, ax=ax)
    assert result is ax
    plt.close("all")


def test_plot_growth_empty():
    port = pd.Series(dtype=float)
    bench = pd.Series(dtype=float)
    ax = plot_growth(port, bench)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_tracking_error_returns_axes(sample_returns):
    port, bench = sample_returns
    te = port - bench
    ax = plot_tracking_error(te)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_weights_history_returns_axes(sample_weights):
    ax = plot_weights_history(sample_weights)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_weights_snapshot_returns_axes(sample_weights):
    ax = plot_weights_snapshot(sample_weights.iloc[-1])
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_performance_summary_basic(sample_returns):
    port, _ = sample_returns
    result = performance_summary(port, steps_per_year=252)
    assert "total_return" in result
    assert "ann_return" in result
    assert "ann_vol" in result
    assert "sharpe" in result
    assert all(isinstance(v, float) for v in result.values())


def test_performance_summary_empty():
    result = performance_summary(pd.Series(dtype=float))
    assert result["total_return"] is None
    assert result["sharpe"] is None
