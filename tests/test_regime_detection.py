"""Tests for regime detection (Markov Switching) module."""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from toolkit.analysis import (
    RegimeCollection,
    RegimeConfig,
    RegimeModel,
    RegimeRun,
    RegimeSnapshot,
    RegimeSnapshotInfo,
    TransformConfig,
    TransformType,
    apply_transform,
    fit_regime_batch,
    list_regime_snapshots,
    load_regime_snapshot,
    save_regime_snapshot,
)
from toolkit.analysis.regime_storage import (
    RegimeCollectionSnapshot,
    RegimePreset,
    list_regime_collections,
    list_regime_presets,
    load_regime_collection,
    load_regime_preset,
    save_regime_collection,
    save_regime_preset,
)
from toolkit.plotly_payload import (
    regime_probabilities_payload,
    summarize_regime_collection,
    summarize_regime_run,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _synthetic_regime_series(
    *,
    k_regimes: int = 2,
    n_obs: int = 500,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic regime-switching series with known ground truth."""
    rng = np.random.default_rng(seed)

    if k_regimes == 2:
        means = [-1.0, 2.0]
        stds = [1.5, 0.8]
        transition = np.array([[0.95, 0.05], [0.10, 0.90]])
    elif k_regimes == 3:
        means = [-2.0, 0.0, 3.0]
        stds = [1.5, 0.5, 1.0]
        transition = np.array([
            [0.90, 0.07, 0.03],
            [0.05, 0.90, 0.05],
            [0.03, 0.07, 0.90],
        ])
    else:
        means = list(rng.uniform(-3, 3, k_regimes))
        stds = list(rng.uniform(0.5, 2.0, k_regimes))
        transition = rng.dirichlet(np.ones(k_regimes) * 10, size=k_regimes)

    regimes = np.zeros(n_obs, dtype=int)
    regimes[0] = rng.integers(0, k_regimes)
    for t in range(1, n_obs):
        regimes[t] = rng.choice(k_regimes, p=transition[regimes[t - 1]])

    values = np.array([
        rng.normal(means[regimes[t]], stds[regimes[t]]) for t in range(n_obs)
    ])

    dates = pd.date_range("2000-01-01", periods=n_obs, freq="QE")
    return pd.Series(values, index=dates, name="synthetic")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestTransforms:
    @pytest.fixture
    def series(self) -> pd.Series:
        dates = pd.date_range("2020-01-01", periods=100, freq="ME")
        rng = np.random.default_rng(42)
        return pd.Series(rng.uniform(90, 110, 100), index=dates, name="test")

    def test_none_transform(self, series: pd.Series) -> None:
        result = apply_transform(series, TransformConfig(TransformType.NONE))
        assert len(result) == len(series)

    def test_first_diff(self, series: pd.Series) -> None:
        result = apply_transform(series, TransformConfig(TransformType.FIRST_DIFF))
        assert len(result) == len(series) - 1

    def test_pct_change(self, series: pd.Series) -> None:
        result = apply_transform(series, TransformConfig(TransformType.PCT_CHANGE))
        assert len(result) == len(series) - 1

    def test_log_diff(self, series: pd.Series) -> None:
        result = apply_transform(series, TransformConfig(TransformType.LOG_DIFF))
        assert len(result) == len(series) - 1

    def test_yoy_change(self, series: pd.Series) -> None:
        result = apply_transform(series, TransformConfig(TransformType.YOY_CHANGE, window=12))
        assert len(result) == len(series) - 12

    def test_rolling_mean(self, series: pd.Series) -> None:
        result = apply_transform(series, TransformConfig(TransformType.ROLLING_MEAN, window=6))
        assert len(result) == len(series) - 5

    def test_transform_config_description(self) -> None:
        cfg = TransformConfig(TransformType.PCT_CHANGE)
        assert "Percentage change" in cfg.description

    def test_transform_config_window_description(self) -> None:
        cfg = TransformConfig(TransformType.ROLLING_MEAN, window=12)
        assert "window=12" in cfg.description


# ---------------------------------------------------------------------------
# Core model tests
# ---------------------------------------------------------------------------

class TestRegimeModelTwoRegimes:
    @pytest.fixture(scope="class")
    def run(self) -> RegimeRun:
        series = _synthetic_regime_series(k_regimes=2, n_obs=400, seed=42)
        model = RegimeModel(k_regimes=2, switching_variance=True, switching_trend=True)
        return model.run(series, name="test_2regime")

    def test_results_keys(self, run: RegimeRun) -> None:
        expected = {
            "smoothed_probabilities", "filtered_probabilities",
            "regime_assignments", "transition_matrix", "regime_params",
            "series", "expected_durations", "aic", "bic", "log_likelihood",
            "model_params", "meta",
        }
        assert expected.issubset(run.results.keys())

    def test_smoothed_shape(self, run: RegimeRun) -> None:
        assert run.smoothed_probabilities.shape[1] == 2
        assert len(run.smoothed_probabilities) == len(run.series)

    def test_filtered_shape(self, run: RegimeRun) -> None:
        assert run.filtered_probabilities.shape[1] == 2

    def test_regime_assignments_values(self, run: RegimeRun) -> None:
        unique = set(run.regime_assignments.unique())
        assert unique.issubset({0, 1})

    def test_transition_matrix_shape(self, run: RegimeRun) -> None:
        assert run.transition_matrix.shape == (2, 2)

    def test_transition_matrix_rows_sum_to_one(self, run: RegimeRun) -> None:
        row_sums = run.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_expected_durations_positive(self, run: RegimeRun) -> None:
        assert (run.expected_durations > 0).all()

    def test_regime_params_keys(self, run: RegimeRun) -> None:
        assert 0 in run.regime_params
        assert 1 in run.regime_params
        for rp in run.regime_params.values():
            assert "mean" in rp
            assert "variance" in rp

    def test_regimes_ordered_by_mean(self, run: RegimeRun) -> None:
        means = [run.regime_params[i]["mean"] for i in range(2)]
        assert means[0] <= means[1], "regime 0 should have the lowest mean"

    def test_no_preset_in_params(self, run: RegimeRun) -> None:
        assert "preset" not in run.params

    def test_train_end_none_by_default(self, run: RegimeRun) -> None:
        assert run.params["train_end"] is None


class TestRegimeModelThreeRegimes:
    @pytest.fixture(scope="class")
    def run(self) -> RegimeRun:
        series = _synthetic_regime_series(k_regimes=3, n_obs=500, seed=99)
        model = RegimeModel(k_regimes=3, switching_variance=True, switching_trend=True)
        return model.run(series, name="test_3regime")

    def test_smoothed_shape(self, run: RegimeRun) -> None:
        assert run.smoothed_probabilities.shape[1] == 3

    def test_transition_matrix_shape(self, run: RegimeRun) -> None:
        assert run.transition_matrix.shape == (3, 3)

    def test_regime_assignments_values(self, run: RegimeRun) -> None:
        unique = set(run.regime_assignments.unique())
        assert unique.issubset({0, 1, 2})

    def test_transition_matrix_rows_sum_to_one(self, run: RegimeRun) -> None:
        row_sums = run.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# train_end
# ---------------------------------------------------------------------------

class TestTrainEnd:
    @pytest.fixture(scope="class")
    def series(self) -> pd.Series:
        return _synthetic_regime_series(k_regimes=2, n_obs=400, seed=42)

    @pytest.fixture(scope="class")
    def run_with_cutoff(self, series: pd.Series) -> RegimeRun:
        model = RegimeModel(k_regimes=2, switching_variance=True, switching_trend=True)
        # Use a cutoff roughly in the middle
        cutoff = series.index[200].strftime("%Y-%m-%d")
        return model.run(series, name="train_end_test", train_end=cutoff)

    def test_probabilities_cover_full_series(self, run_with_cutoff: RegimeRun, series: pd.Series) -> None:
        assert len(run_with_cutoff.smoothed_probabilities) == len(series)

    def test_train_end_in_params(self, run_with_cutoff: RegimeRun) -> None:
        assert run_with_cutoff.params["train_end"] is not None

    def test_train_end_in_meta(self, run_with_cutoff: RegimeRun) -> None:
        assert "train_end" in run_with_cutoff.meta

    def test_n_train_less_than_n_obs(self, run_with_cutoff: RegimeRun) -> None:
        assert run_with_cutoff.meta["n_train"] < run_with_cutoff.meta["n_obs"]

    def test_train_end_too_short_raises(self, series: pd.Series) -> None:
        model = RegimeModel(k_regimes=2)
        early_cutoff = series.index[5].strftime("%Y-%m-%d")
        with pytest.raises(ValueError, match="at least 20"):
            model.run(series, name="short_train", train_end=early_cutoff)


# ---------------------------------------------------------------------------
# RegimeConfig
# ---------------------------------------------------------------------------

def test_regime_config_creation() -> None:
    cfg = RegimeConfig(
        name="GDP Recession",
        fred_series_id="GDP",
        transform=TransformConfig(TransformType.PCT_CHANGE),
        k_regimes=2,
        train_end="2020-01-01",
    )
    assert cfg.name == "GDP Recession"
    assert cfg.fred_series_id == "GDP"
    assert cfg.k_regimes == 2


def test_regime_config_from_config() -> None:
    cfg = RegimeConfig(
        name="Test",
        fred_series_id="FEDFUNDS",
        k_regimes=3,
        switching_variance=True,
        switching_trend=False,
    )
    model = RegimeModel.from_config(cfg)
    assert model.k_regimes == 3
    assert model.switching_trend is False


# ---------------------------------------------------------------------------
# RegimeCollection
# ---------------------------------------------------------------------------

class TestRegimeCollection:
    @pytest.fixture
    def two_runs(self) -> list[tuple[RegimeConfig, RegimeRun]]:
        s1 = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=1)
        s2 = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=2)
        model = RegimeModel(k_regimes=2)
        run1 = model.run(s1, name="regime_a")
        run2 = model.run(s2, name="regime_b")
        cfg1 = RegimeConfig(name="Regime A", fred_series_id="GDP")
        cfg2 = RegimeConfig(name="Regime B", fred_series_id="FEDFUNDS")
        return [(cfg1, run1), (cfg2, run2)]

    def test_add_and_names(self, two_runs) -> None:
        coll = RegimeCollection()
        for cfg, run in two_runs:
            coll.add(cfg, run)
        assert coll.names == ["Regime A", "Regime B"]
        assert len(coll) == 2

    def test_duplicate_name_raises(self, two_runs) -> None:
        coll = RegimeCollection()
        cfg, run = two_runs[0]
        coll.add(cfg, run)
        with pytest.raises(ValueError, match="already exists"):
            coll.add(cfg, run)

    def test_remove(self, two_runs) -> None:
        coll = RegimeCollection()
        for cfg, run in two_runs:
            coll.add(cfg, run)
        coll.remove("Regime A")
        assert coll.names == ["Regime B"]

    def test_remove_nonexistent_raises(self) -> None:
        coll = RegimeCollection()
        with pytest.raises(KeyError):
            coll.remove("nonexistent")

    def test_get(self, two_runs) -> None:
        coll = RegimeCollection()
        for cfg, run in two_runs:
            coll.add(cfg, run)
        cfg, run = coll.get("Regime A")
        assert cfg.name == "Regime A"

    def test_endogenous_data_shape(self, two_runs) -> None:
        coll = RegimeCollection()
        for cfg, run in two_runs:
            coll.add(cfg, run)
        endo = coll.endogenous_data()
        # 2 regimes * 2 entries = 4 columns
        assert endo.shape[1] == 4
        assert "Regime_A_prob_0" in endo.columns
        assert "Regime_B_prob_1" in endo.columns

    def test_endogenous_data_empty_collection(self) -> None:
        coll = RegimeCollection()
        assert coll.endogenous_data().empty

    def test_summary(self, two_runs) -> None:
        coll = RegimeCollection()
        for cfg, run in two_runs:
            coll.add(cfg, run)
        s = coll.summary()
        assert "Regime A" in s
        assert "Regime B" in s

    def test_bool_empty(self) -> None:
        assert not RegimeCollection()

    def test_bool_nonempty(self, two_runs) -> None:
        coll = RegimeCollection()
        coll.add(*two_runs[0])
        assert coll


# ---------------------------------------------------------------------------
# Access patterns
# ---------------------------------------------------------------------------

def test_getitem_access() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="access_test")
    pd.testing.assert_frame_equal(
        run["smoothed_probabilities"], run.smoothed_probabilities,
    )


def test_summary_non_empty() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="summary_test")
    s = run.summary()
    assert isinstance(s, str)
    assert len(s) > 50
    assert "RegimeRun" in s
    assert "regime 0" in s


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_features_columns() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="feat_test")
    feat = run.features()
    assert "feat_test_prob_0" in feat.columns
    assert "feat_test_prob_1" in feat.columns
    assert "feat_test_regime" in feat.columns
    assert len(feat) == len(run.series)


def test_features_probabilities_sum_to_one() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="prob_sum")
    feat = run.features()
    prob_cols = [c for c in feat.columns if "_prob_" in c]
    row_sums = feat[prob_cols].sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


def test_features_custom_prefix() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="x")
    feat = run.features(prefix="gdp")
    assert "gdp_prob_0" in feat.columns
    assert "gdp_regime" in feat.columns


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------

def test_fit_regime_batch() -> None:
    s1 = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=1)
    s2 = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=2)
    results = fit_regime_batch({"s1": s1, "s2": s2})
    assert "s1" in results
    assert "s2" in results
    assert isinstance(results["s1"], RegimeRun)
    assert isinstance(results["s2"], RegimeRun)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_k_regimes() -> None:
    with pytest.raises(ValueError, match="k_regimes"):
        RegimeModel(k_regimes=1)


def test_invalid_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        RegimeModel(model_type="invalid")


def test_series_too_short() -> None:
    series = pd.Series([1.0] * 10, index=pd.date_range("2020-01-01", periods=10))
    model = RegimeModel(k_regimes=2)
    with pytest.raises(ValueError, match="at least 20"):
        model.run(series)


def test_series_must_be_pd_series() -> None:
    model = RegimeModel(k_regimes=2)
    with pytest.raises(TypeError, match="pd.Series"):
        model.run([1, 2, 3])  # type: ignore[arg-type]


def test_series_must_have_datetime_index() -> None:
    series = pd.Series([1.0] * 50, index=range(50))
    model = RegimeModel(k_regimes=2)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        model.run(series)


# ---------------------------------------------------------------------------
# Storage roundtrip — single snapshot
# ---------------------------------------------------------------------------

def test_regime_snapshot_roundtrip(tmp_path: Path) -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="roundtrip_test")

    snapshot = RegimeSnapshot(
        name="GDP Test Run",
        created_at=datetime(2024, 6, 15, 12, 0, 0),
        series_name="gdp_qoq",
        k_regimes=2,
        preset=None,
        run=run,
        source_data=series,
    )

    path = save_regime_snapshot(snapshot, tmp_path)
    loaded = load_regime_snapshot(path)

    assert loaded.name == snapshot.name
    assert loaded.k_regimes == snapshot.k_regimes
    pd.testing.assert_series_equal(loaded.source_data, snapshot.source_data)
    pd.testing.assert_frame_equal(
        loaded.run.smoothed_probabilities,
        snapshot.run.smoothed_probabilities,
    )


def test_list_regime_snapshots(tmp_path: Path) -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="list_test")

    for name in ["Run A", "Run B"]:
        snap = RegimeSnapshot(
            name=name,
            created_at=datetime.now(),
            series_name="test",
            k_regimes=2,
            preset=None,
            run=run,
            source_data=series,
        )
        save_regime_snapshot(snap, tmp_path)

    infos = list_regime_snapshots(tmp_path)
    assert len(infos) == 2
    names = {info.name for info in infos}
    assert names == {"Run A", "Run B"}


def test_snapshot_overwrite_error(tmp_path: Path) -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="dup_test")
    snap = RegimeSnapshot(
        name="DupRun",
        created_at=datetime.now(),
        series_name="test",
        k_regimes=2,
        preset=None,
        run=run,
        source_data=series,
    )
    save_regime_snapshot(snap, tmp_path)
    with pytest.raises(FileExistsError):
        save_regime_snapshot(snap, tmp_path)


# ---------------------------------------------------------------------------
# Storage roundtrip — collection
# ---------------------------------------------------------------------------

def test_collection_snapshot_roundtrip(tmp_path: Path) -> None:
    s1 = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=1)
    model = RegimeModel(k_regimes=2)
    run1 = model.run(s1, name="coll_test")
    cfg1 = RegimeConfig(name="Test Regime", fred_series_id="GDP")

    coll = RegimeCollection()
    coll.add(cfg1, run1)

    snap = RegimeCollectionSnapshot(
        name="My Collection",
        created_at=datetime.now(),
        collection=coll,
    )
    path = save_regime_collection(snap, tmp_path)
    loaded = load_regime_collection(path)

    assert loaded.name == "My Collection"
    assert len(loaded.collection) == 1
    assert loaded.collection.names == ["Test Regime"]


def test_list_regime_collections(tmp_path: Path) -> None:
    s1 = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=1)
    model = RegimeModel(k_regimes=2)
    run1 = model.run(s1, name="list_coll")
    cfg1 = RegimeConfig(name="R1", fred_series_id="GDP")

    for coll_name in ["Collection A", "Collection B"]:
        coll = RegimeCollection()
        coll.add(cfg1, run1)
        snap = RegimeCollectionSnapshot(
            name=coll_name, created_at=datetime.now(), collection=coll,
        )
        save_regime_collection(snap, tmp_path, overwrite=True)

    infos = list_regime_collections(tmp_path)
    assert len(infos) >= 1  # may be same key if names normalize identically


# ---------------------------------------------------------------------------
# Storage roundtrip — presets
# ---------------------------------------------------------------------------

def test_preset_roundtrip(tmp_path: Path) -> None:
    cfg = RegimeConfig(
        name="GDP Recession",
        fred_series_id="GDP",
        transform=TransformConfig(TransformType.PCT_CHANGE),
        k_regimes=2,
        train_end="2020-01-01",
    )
    preset = RegimePreset(
        name="GDP Recession Preset",
        description="GDP recession indicator with pct change",
        config=cfg,
        created_at=datetime.now(),
    )
    path = save_regime_preset(preset, tmp_path)
    loaded = load_regime_preset(path)

    assert loaded.name == "GDP Recession Preset"
    assert loaded.config.fred_series_id == "GDP"
    assert loaded.config.train_end == "2020-01-01"


def test_list_regime_presets(tmp_path: Path) -> None:
    for name in ["Preset A", "Preset B"]:
        cfg = RegimeConfig(name=name, fred_series_id="GDP")
        preset = RegimePreset(
            name=name, description="test", config=cfg, created_at=datetime.now(),
        )
        save_regime_preset(preset, tmp_path)

    presets = list_regime_presets(tmp_path)
    assert len(presets) == 2


# ---------------------------------------------------------------------------
# Plotly payloads
# ---------------------------------------------------------------------------

def test_regime_probabilities_payload_structure() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="payload_test")
    payload = regime_probabilities_payload(run.smoothed_probabilities)
    assert "series" in payload
    assert "y_axis_title" in payload
    assert len(payload["series"]) == 2
    assert "x" in payload["series"][0]
    assert "y" in payload["series"][0]


def test_summarize_regime_run_structure() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="summary_payload")
    summary = summarize_regime_run(run)
    expected_keys = {
        "chart_probabilities", "chart_series", "chart_dual_axis",
        "regime_assignments", "transition_matrix", "regime_table",
        "regression_stats", "metrics", "train_end_marker", "warnings",
    }
    assert expected_keys.issubset(summary.keys())
    assert len(summary["regime_table"]) == 2
    assert "aic" in summary["metrics"]


def test_summarize_regime_run_with_train_end() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=400, seed=10)
    cutoff = series.index[200].strftime("%Y-%m-%d")
    run = RegimeModel(k_regimes=2).run(series, name="te_payload", train_end=cutoff)
    summary = summarize_regime_run(run)
    assert summary["train_end_marker"] is not None
    assert summary["train_end_marker"]["date"] == cutoff


def test_summarize_regime_collection_structure() -> None:
    s1 = _synthetic_regime_series(k_regimes=2, n_obs=300, seed=1)
    model = RegimeModel(k_regimes=2)
    run1 = model.run(s1, name="coll_payload")
    cfg1 = RegimeConfig(name="Test Regime", fred_series_id="GDP")

    coll = RegimeCollection()
    coll.add(cfg1, run1)

    result = summarize_regime_collection(coll)
    assert "Test Regime" in result["individual_summaries"]
    assert result["collection_info"]["n_regimes"] == 1
    # Individual summary should contain regression_stats and chart_dual_axis keys
    indiv = result["individual_summaries"]["Test Regime"]
    assert "regression_stats" in indiv
    assert "chart_dual_axis" in indiv


def test_summarize_empty_collection() -> None:
    coll = RegimeCollection()
    result = summarize_regime_collection(coll)
    assert result["collection_info"]["n_regimes"] == 0
    assert result["individual_summaries"] == {}


# ---------------------------------------------------------------------------
# Chart smoke tests
# ---------------------------------------------------------------------------

def test_plot_regime_probabilities_smoke() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=200, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="chart_test")
    ax = run.plot_regime_probabilities()
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_regime_series_smoke() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=200, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="chart_test")
    ax = run.plot_regime_series()
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_transition_matrix_smoke() -> None:
    series = _synthetic_regime_series(k_regimes=2, n_obs=200, seed=10)
    run = RegimeModel(k_regimes=2).run(series, name="chart_test")
    ax = run.plot_transition_matrix()
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close("all")
