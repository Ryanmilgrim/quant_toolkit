"""Run a style analysis on synthetic data and generate charts."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_toolkit.analysis import StyleAnalysis

# Build a synthetic monthly universe with known weights
rng = np.random.default_rng(0)
periods, n_assets = 60, 12
dates = pd.date_range("2015-01-31", periods=periods, freq="ME")
asset_cols = [f"Industry_{i}" for i in range(n_assets)]

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
uni = pd.concat([benchmark, assets], axis=1, keys=["benchmarks", "assets"])

# Run the analysis
sa = StyleAnalysis(uni, benchmark_name="Mkt")
run = sa.run(style_window=12, optimize_frequency="monthly")

print(run.summary())
print()

# Print performance metrics
perf = run.performance()
for section, metrics in perf.items():
    print(f"{section}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()

# Generate chart panel
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
run.plot_growth(ax=axes[0, 0])
run.plot_tracking_error(ax=axes[0, 1])
run.plot_weights(ax=axes[1, 0])
run.plot_weights_snapshot(ax=axes[1, 1])
fig.tight_layout()
fig.savefig("style_analysis_charts.png", dpi=150)
print("Saved style_analysis_charts.png")
