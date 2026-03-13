from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from toolkit.analysis import (
    StyleAnalysis,
    StyleAnalysisSnapshot,
    list_style_snapshots,
    load_style_snapshot,
    normalize_snapshot_name,
    save_style_snapshot,
)


def _synthetic_monthly_universe(
    *,
    periods: int = 36,
    n_assets: int = 8,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=periods, freq="ME")
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


def test_style_snapshot_roundtrip(tmp_path: Path) -> None:
    df = _synthetic_monthly_universe()
    run = StyleAnalysis(df, benchmark_name="Mkt").run(
        style_window=12,
        optimize_frequency="monthly",
    )

    snapshot = StyleAnalysisSnapshot(
        name="Market Portfolio Estimate",
        created_at=datetime(2024, 1, 1, 9, 0, 0),
        universe=10,
        weighting="value",
        factor_set="ff3",
        start_date=date(2018, 1, 31),
        end_date=None,
        run=run,
        universe_data=df,
    )

    path = save_style_snapshot(snapshot, tmp_path)
    loaded = load_style_snapshot(path)

    assert loaded.name == snapshot.name
    assert loaded.universe == snapshot.universe
    pd.testing.assert_frame_equal(loaded.universe_data, snapshot.universe_data)
    pd.testing.assert_frame_equal(loaded.run.weights, snapshot.run.weights)

    infos = list_style_snapshots(tmp_path)
    assert len(infos) == 1
    assert infos[0].name == snapshot.name
    assert infos[0].key == normalize_snapshot_name(snapshot.name)
    assert infos[0].window == snapshot.run.params.get("style_window")
    assert infos[0].rebalance == snapshot.run.params.get("optimize_frequency")
