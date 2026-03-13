import numpy as np
import pandas as pd

from toolkit.plotly_payload import (
    line_chart_payload,
    summarize_style_run,
    weights_history_payload,
)
from toolkit.analysis import StyleAnalysis


def test_line_chart_payload_basic():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}, index=dates, dtype=float)
    result = line_chart_payload(df, y_axis_title="Test")
    assert result["y_axis_title"] == "Test"
    assert len(result["series"]) == 2
    assert result["series"][0]["name"] == "A"
    assert len(result["series"][0]["x"]) == 5


def test_line_chart_payload_empty():
    df = pd.DataFrame()
    result = line_chart_payload(df, y_axis_title="Empty")
    assert result["series"] == []


def test_weights_history_payload_basic():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    w = pd.DataFrame({"X": [0.6, 0.5, 0.7], "Y": [0.4, 0.5, 0.3]}, index=dates)
    result = weights_history_payload(w)
    assert result["y_axis_title"] == "Weight (%)"
    assert len(result["series"]) == 2
    # Weights should sum to 100% per row
    for i in range(3):
        total = sum(s["y"][i] for s in result["series"])
        assert abs(total - 100.0) < 0.1


def test_weights_history_payload_empty():
    result = weights_history_payload(pd.DataFrame())
    assert result["series"] == []


def _synthetic_universe(periods=48, n_assets=10, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=periods, freq="ME")
    asset_cols = [f"A{i}" for i in range(n_assets)]
    assets = pd.DataFrame(
        rng.normal(loc=0.002, scale=0.01, size=(periods, n_assets)),
        index=dates, columns=asset_cols,
    )
    w_true = np.zeros(n_assets)
    w_true[:3] = [0.5, 0.3, 0.2]
    benchmark = pd.Series(
        assets.to_numpy() @ w_true + rng.normal(0, 0.001, periods),
        index=dates, name="Mkt",
    )
    return pd.concat([benchmark, assets], axis=1, keys=["benchmarks", "assets"])


def test_summarize_style_run():
    df = _synthetic_universe()
    run = StyleAnalysis(df, benchmark_name="Mkt").run(
        style_window=12, optimize_frequency="monthly",
    )
    result = summarize_style_run(run)
    assert "chart_growth" in result
    assert "chart_tracking" in result
    assert "weights_history" in result
    assert "weights_table" in result
    assert "metrics" in result
    assert isinstance(result["warnings"], list)
    assert result["chart_growth"] is not None
    assert len(result["chart_growth"]["series"]) == 2
