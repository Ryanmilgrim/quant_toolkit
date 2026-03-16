"""Regime Detection — example usage.

Demonstrates the iterative regime-building workflow:
1. Generate synthetic data (simulating FRED series)
2. Apply transformations
3. Fit Markov Switching models with training cutoff dates
4. Build a RegimeCollection iteratively
5. Inspect the combined endogenous data
6. Save/load collections and presets
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolkit.analysis import (
    RegimeCollection,
    RegimeConfig,
    RegimeModel,
    RegimeRun,
    TransformConfig,
    TransformType,
    apply_transform,
    fit_regime_batch,
)
from toolkit.analysis.regime_storage import (
    RegimeCollectionSnapshot,
    RegimePreset,
    list_regime_collections,
    list_regime_presets,
    load_regime_collection,
    save_regime_collection,
    save_regime_preset,
)


# ---------------------------------------------------------------------------
# 1. Generate synthetic data (simulating FRED series)
# ---------------------------------------------------------------------------

def make_gdp_series(seed: int = 42, n_obs: int = 200) -> pd.Series:
    """Synthetic GDP QoQ growth with 2 regimes (expansion vs recession)."""
    rng = np.random.default_rng(seed)
    transition = np.array([[0.95, 0.05], [0.10, 0.90]])
    means = [2.5, -1.0]
    stds = [0.8, 1.5]
    regimes = np.zeros(n_obs, dtype=int)
    regimes[0] = 0
    for t in range(1, n_obs):
        regimes[t] = rng.choice(2, p=transition[regimes[t - 1]])
    values = np.array([
        rng.normal(means[regimes[t]], stds[regimes[t]]) for t in range(n_obs)
    ])
    dates = pd.date_range("1970-03-31", periods=n_obs, freq="QE")
    return pd.Series(values, index=dates, name="GDP_QoQ")


def make_fed_funds_series(seed: int = 99, n_obs: int = 300) -> pd.Series:
    """Synthetic Fed Funds rate changes with 3 regimes."""
    rng = np.random.default_rng(seed)
    transition = np.array([
        [0.92, 0.05, 0.03],
        [0.04, 0.92, 0.04],
        [0.03, 0.05, 0.92],
    ])
    means = [-0.25, 0.0, 0.25]
    stds = [0.10, 0.05, 0.10]
    regimes = np.zeros(n_obs, dtype=int)
    regimes[0] = 1
    for t in range(1, n_obs):
        regimes[t] = rng.choice(3, p=transition[regimes[t - 1]])
    values = np.array([
        rng.normal(means[regimes[t]], stds[regimes[t]]) for t in range(n_obs)
    ])
    dates = pd.date_range("1990-01-31", periods=n_obs, freq="ME")
    return pd.Series(values, index=dates, name="FedFunds_Change")


def make_yield_curve_series(seed: int = 77, n_obs: int = 250) -> pd.Series:
    """Synthetic yield curve slope (10Y-2Y spread) with 2 regimes."""
    rng = np.random.default_rng(seed)
    transition = np.array([[0.97, 0.03], [0.06, 0.94]])
    means = [1.5, -0.3]
    stds = [0.4, 0.6]
    regimes = np.zeros(n_obs, dtype=int)
    regimes[0] = 0
    for t in range(1, n_obs):
        regimes[t] = rng.choice(2, p=transition[regimes[t - 1]])
    values = np.array([
        rng.normal(means[regimes[t]], stds[regimes[t]]) for t in range(n_obs)
    ])
    dates = pd.date_range("1985-01-31", periods=n_obs, freq="ME")
    return pd.Series(values, index=dates, name="T10Y2Y")


# ---------------------------------------------------------------------------
# 2. Apply transformations
# ---------------------------------------------------------------------------

print("=" * 60)
print("REGIME DETECTION — ITERATIVE BUILDER EXAMPLE")
print("=" * 60)

gdp_raw = make_gdp_series()
ffr_raw = make_fed_funds_series()
yc_raw = make_yield_curve_series()

# GDP: use as-is (already QoQ growth)
gdp_transform = TransformConfig(transform=TransformType.NONE)
gdp_series = apply_transform(gdp_raw, gdp_transform)

# Fed Funds: rolling mean to smooth
ffr_transform = TransformConfig(transform=TransformType.ROLLING_MEAN, window=6)
ffr_series = apply_transform(ffr_raw, ffr_transform)

# Yield curve: first difference
yc_transform = TransformConfig(transform=TransformType.FIRST_DIFF)
yc_series = apply_transform(yc_raw, yc_transform)

print(f"\nGDP series: {len(gdp_series)} obs, transform: {gdp_transform.description}")
print(f"Fed Funds series: {len(ffr_series)} obs, transform: {ffr_transform.description}")
print(f"Yield Curve series: {len(yc_series)} obs, transform: {yc_transform.description}")


# ---------------------------------------------------------------------------
# 3. Fit models with training cutoff dates
# ---------------------------------------------------------------------------

print("\n--- Fitting models with train_end ---")

# GDP Recession Indicator (2 regimes, train up to 2010)
gdp_config = RegimeConfig(
    name="GDP Recession",
    fred_series_id="GDP",
    transform=gdp_transform,
    description="Expansion vs recession based on GDP QoQ growth",
    k_regimes=2,
    switching_variance=True,
    switching_trend=True,
    train_end="2010-01-01",
    regime_labels={0: "Expansion", 1: "Recession"},
)
gdp_model = RegimeModel.from_config(gdp_config)
gdp_run = gdp_model.run(gdp_series, name="GDP Recession", train_end="2010-01-01")
print(f"\nGDP Recession:")
print(f"  Total obs: {gdp_run.meta['n_obs']}, Training obs: {gdp_run.meta['n_train']}")
print(f"  Train end: {gdp_run.meta['train_end']}")
print(gdp_run.summary())

# Fed Funds Regime (3 regimes, train up to 2010)
ffr_config = RegimeConfig(
    name="Fed Funds Regime",
    fred_series_id="FEDFUNDS",
    transform=ffr_transform,
    description="Easing / neutral / tightening from Fed Funds rate",
    k_regimes=3,
    switching_variance=True,
    switching_trend=True,
    train_end="2010-01-01",
    regime_labels={0: "Easing", 1: "Neutral", 2: "Tightening"},
)
ffr_model = RegimeModel.from_config(ffr_config)
ffr_run = ffr_model.run(ffr_series, name="Fed Funds Regime", train_end="2010-01-01")
print(f"\nFed Funds Regime:")
print(f"  Total obs: {ffr_run.meta['n_obs']}, Training obs: {ffr_run.meta['n_train']}")
print(ffr_run.summary())

# Yield Curve (2 regimes, no train_end — full sample)
yc_config = RegimeConfig(
    name="Yield Curve Slope",
    fred_series_id="T10Y2Y",
    transform=yc_transform,
    description="Normal vs inverted yield curve slope changes",
    k_regimes=2,
    switching_variance=True,
    switching_trend=False,
    regime_labels={0: "Normal", 1: "Inverted"},
)
yc_model = RegimeModel.from_config(yc_config)
yc_run = yc_model.run(yc_series, name="Yield Curve Slope")
print(f"\nYield Curve Slope:")
print(f"  Total obs: {yc_run.meta['n_obs']}")
print(yc_run.summary())


# ---------------------------------------------------------------------------
# 4. Build a RegimeCollection iteratively
# ---------------------------------------------------------------------------

print("\n--- Building RegimeCollection ---")

collection = RegimeCollection()

collection.add(gdp_config, gdp_run)
print(f"Added '{gdp_config.name}' → {len(collection)} regime(s)")

collection.add(ffr_config, ffr_run)
print(f"Added '{ffr_config.name}' → {len(collection)} regime(s)")

collection.add(yc_config, yc_run)
print(f"Added '{yc_config.name}' → {len(collection)} regime(s)")

print(f"\nCollection names: {collection.names}")
print(collection.summary())


# ---------------------------------------------------------------------------
# 5. Inspect the combined endogenous data
# ---------------------------------------------------------------------------

print("\n--- Endogenous Data (combined soft probabilities) ---")

endo = collection.endogenous_data()
print(f"Shape: {endo.shape}")
print(f"Columns: {list(endo.columns)}")
print(f"Date range: {endo.index.min()} → {endo.index.max()}")
print(f"\nFirst 5 rows:")
print(endo.head().to_string())
print(f"\nLast 5 rows:")
print(endo.tail().to_string())


# ---------------------------------------------------------------------------
# 6. Extract ML features
# ---------------------------------------------------------------------------

print("\n--- ML Features ---")
features = pd.concat(
    [gdp_run.features(), ffr_run.features(), yc_run.features()],
    axis=1,
)
print(f"Shape: {features.shape}")
print(f"Columns: {list(features.columns)}")
print(features.head(10).to_string())


# ---------------------------------------------------------------------------
# 7. Charts
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

gdp_run.plot_regime_probabilities(ax=axes[0, 0], title="GDP: Regime Probabilities")
gdp_run.plot_regime_series(ax=axes[0, 1], title="GDP: Series with Regimes")
gdp_run.plot_transition_matrix(ax=axes[0, 2], title="GDP: Transition Matrix")

ffr_run.plot_regime_probabilities(ax=axes[1, 0], title="Fed Funds: Regime Probabilities")
ffr_run.plot_regime_series(ax=axes[1, 1], title="Fed Funds: Series with Regimes")
yc_run.plot_regime_probabilities(ax=axes[1, 2], title="Yield Curve: Regime Probabilities")

plt.tight_layout()
plt.savefig("regime_detection_charts.png", dpi=150)
print("\nSaved charts to regime_detection_charts.png")


# ---------------------------------------------------------------------------
# 8. Batch fitting
# ---------------------------------------------------------------------------

print("\n--- Batch Fitting ---")
batch_results = fit_regime_batch({"gdp": gdp_series, "ffr": ffr_series})
print(f"Fitted {len(batch_results)} series: {list(batch_results.keys())}")


# ---------------------------------------------------------------------------
# 9. Save / load collection roundtrip
# ---------------------------------------------------------------------------

print("\n--- Collection Save/Load Roundtrip ---")

tmp_dir = Path("_regime_example_output")
tmp_dir.mkdir(exist_ok=True)

snap = RegimeCollectionSnapshot(
    name="Example Collection",
    created_at=datetime.now(),
    collection=collection,
)
save_regime_collection(snap, tmp_dir, overwrite=True)
print(f"Saved collection '{snap.name}'")

infos = list_regime_collections(tmp_dir)
print(f"Listed {len(infos)} collection(s): {[i.name for i in infos]}")

# Load it back
from toolkit.analysis.style_storage import snapshot_path
coll_path = snapshot_path(tmp_dir / "collections", infos[0].key)
loaded = load_regime_collection(coll_path)
print(f"Loaded collection: '{loaded.name}' with {len(loaded.collection)} regime(s)")
print(f"  Regime names: {loaded.collection.names}")


# ---------------------------------------------------------------------------
# 10. Preset save / load roundtrip
# ---------------------------------------------------------------------------

print("\n--- Preset Save/Load Roundtrip ---")

preset = RegimePreset(
    name="GDP Recession Default",
    description="2-regime GDP QoQ growth, switching variance and trend",
    config=gdp_config,
    created_at=datetime.now(),
)
save_regime_preset(preset, tmp_dir, overwrite=True)
print(f"Saved preset '{preset.name}'")

presets = list_regime_presets(tmp_dir)
print(f"Listed {len(presets)} preset(s): {[p.name for p in presets]}")


# ---------------------------------------------------------------------------
# Clean up
# ---------------------------------------------------------------------------

import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
if Path("regime_detection_charts.png").exists():
    Path("regime_detection_charts.png").unlink()

print("\nDone!")
