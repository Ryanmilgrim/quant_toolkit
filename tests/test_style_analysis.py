import numpy as np
import pandas as pd

from toolkit.analysis import StyleAnalysis


def _synthetic_monthly_universe(
    *,
    periods: int = 60,
    n_assets: int = 12,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=periods, freq="ME")
    asset_cols = [f"A{i}" for i in range(n_assets)]
    assets = pd.DataFrame(
        rng.normal(loc=0.002, scale=0.01, size=(periods, n_assets)),
        index=dates,
        columns=asset_cols,
    )

    w_true = np.zeros(n_assets)
    w_true[:3] = np.array([0.5, 0.3, 0.2])
    benchmark = pd.Series(
        assets.to_numpy() @ w_true + rng.normal(loc=0.0, scale=0.001, size=periods),
        index=dates,
        name="Mkt",
    )
    return pd.concat([benchmark, assets], axis=1, keys=["benchmarks", "assets"])


def test_style_analysis_least_squares_outputs_are_well_formed() -> None:
    df = _synthetic_monthly_universe(periods=48, n_assets=10, seed=1)

    run = StyleAnalysis(df, benchmark_name="Mkt").run(
        style_window=12,
        optimize_frequency="monthly",
    )

    w = run.weights
    assert not w.empty
    assert np.allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-10)
    assert (w.to_numpy() >= -1e-12).all()

    te = run.tracking_error.dropna()
    assert len(te) == len(w)
    assert np.isfinite(te.to_numpy()).all()


def test_style_analysis_missing_asset_forces_zero_weight() -> None:
    window = 12
    df = _synthetic_monthly_universe(periods=60, n_assets=12, seed=2)

    nan_date = df.index[10]
    df.loc[nan_date, ("assets", "A7")] = np.nan

    run = StyleAnalysis(df, benchmark_name="Mkt").run(
        style_window=window,
        optimize_frequency="monthly",
    )
    w = run.weights
    assert not w.empty

    i_nan = int(np.where(df.index == nan_date)[0][0])
    i_start = max(i_nan, window - 1)
    i_end = min(i_nan + window - 1, len(df.index) - 1)
    affected_dates = df.index[i_start : i_end + 1]

    assert (w.loc[affected_dates, "A7"] == 0.0).all()

