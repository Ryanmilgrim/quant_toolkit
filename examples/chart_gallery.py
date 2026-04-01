"""Chart Gallery -- example usage.

Demonstrates:
  - All matplotlib chart functions for style analysis
  - Individual chart calls (growth, tracking error, weights, snapshot)

Run with::

    python examples/chart_gallery.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolkit.charts import (
    plot_growth,
    plot_tracking_error,
    plot_weights_history,
    plot_weights_snapshot,
    performance_summary,
)

rng = np.random.default_rng(42)
dates = pd.date_range("2020-01-01", periods=500, freq="B")

portfolio = pd.Series(rng.normal(0.0004, 0.012, 500), index=dates)
benchmark = pd.Series(rng.normal(0.0003, 0.010, 500), index=dates)
tracking = portfolio - benchmark

weight_dates = pd.date_range("2020-01-31", periods=24, freq="ME")
weights = pd.DataFrame(
    rng.dirichlet(np.ones(5), size=24),
    index=weight_dates,
    columns=["Growth", "Value", "Quality", "Momentum", "Size"],
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_growth(portfolio, benchmark, ax=axes[0, 0])
plot_tracking_error(tracking, ax=axes[0, 1])
plot_weights_history(weights, ax=axes[1, 0])
plot_weights_snapshot(weights.iloc[-1], ax=axes[1, 1])

fig.suptitle("Chart Gallery", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("chart_gallery.png", dpi=150)
print("Saved chart_gallery.png")

# Performance summary
perf = performance_summary(portfolio)
print("\nPortfolio Performance:")
for k, v in perf.items():
    print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")
