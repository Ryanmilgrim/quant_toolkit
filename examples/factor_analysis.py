"""Factor analysis example — synthetic data demo.

Run with::

    python examples/factor_analysis.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolkit.analysis.risk_model import RiskModel, RiskModelRun, FactorModel, FactorRun


def main() -> None:
    # --- Build synthetic data ---
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=700)

    n_assets = 25
    n_factors = 5

    factors = pd.DataFrame(
        rng.normal(0, 0.007, size=(len(dates), n_factors)),
        index=dates,
        columns=[f"F{i + 1}" for i in range(n_factors)],
    )
    B_true = rng.normal(0, 0.8, size=(n_assets, n_factors))
    resid = rng.normal(0, 0.01, size=(len(dates), n_assets))

    assets_excess = factors.to_numpy() @ B_true.T + resid
    rf = pd.Series(
        rng.normal(0.00005, 0.0001, size=len(dates)),
        index=dates, name="Rf",
    )
    assets = (
        pd.DataFrame(
            assets_excess, index=dates,
            columns=[f"A{i + 1}" for i in range(n_assets)],
        )
        + rf.values.reshape(-1, 1)
    )

    # --- Fit and evaluate ---
    fm = FactorModel(rf_name="Rf", garch_dist="t", pca_demean=False)
    run = fm.evaluate_train_test(
        assets=assets, factors=factors, rf=rf,
        train_fraction=0.7, realized_window=60, progress=True,
    )

    # --- Sanity checks ---
    assert isinstance(run, FactorRun)
    assert "beta_loadings" in run.results
    assert "asset_cov_forecast" in run.results
    assert "evaluation" in run.results

    # --- Print summary ---
    print(run.summary())

    ev = run["evaluation"]
    print("\nEvaluation summary:")
    train_end = ev.get("params", {}).get("train_end")
    if train_end is not None:
        print(f"  train_end: {train_end}")
    for k, v in ev.get("summary", {}).items():
        print(f"  {k}: {v}")

    # --- Generate chart panel ---
    asset0 = str(run.beta_loadings.index[0])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    run.plot_beta_heatmap(ax=axes[0, 0])
    run.plot_factor_risk_heatmap(metric="correlation", ax=axes[0, 1])
    run.plot_volatility_backtest(asset=asset0, ax=axes[1, 0])
    run.plot_returns_with_confidence_bands(asset=asset0, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig("factor_analysis_charts.png", dpi=150)
    print("\nSaved factor_analysis_charts.png")


if __name__ == "__main__":
    main()
