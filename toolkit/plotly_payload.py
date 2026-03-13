"""Plotly.js payload builders for web applications.

These functions convert pandas DataFrames into JSON-serialisable dicts
suitable for rendering with Plotly.js on the client side.  They are a
bridge between :mod:`quant_toolkit` analytics and a web presentation layer.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from toolkit.charts import STEPS_PER_YEAR, _top_weights, performance_summary


def line_chart_payload(
    df: pd.DataFrame,
    *,
    y_axis_title: str,
    round_to: int = 6,
) -> dict[str, object]:
    """Convert a DataFrame to a Plotly-friendly line chart payload."""
    if df.empty:
        return {"series": [], "y_axis_title": y_axis_title}

    dates = [dt.strftime("%Y-%m-%d") for dt in df.index]
    series = [
        {"name": col, "x": dates, "y": df[col].round(round_to).tolist()}
        for col in df.columns
    ]
    return {"series": series, "y_axis_title": y_axis_title}


def weights_history_payload(
    weights: pd.DataFrame,
) -> dict[str, object]:
    """Convert a weights DataFrame to a stacked-area Plotly payload."""
    if weights.empty:
        return {"series": [], "y_axis_title": "Weight (%)"}

    data = weights.copy().astype(float).clip(lower=0.0)
    row_sum = data.sum(axis=1).replace(0.0, np.nan)
    data = data.div(row_sum, axis=0).fillna(0.0).mul(100.0)

    dates = [dt.strftime("%Y-%m-%d") for dt in data.index]
    series = [
        {"name": col, "x": dates, "y": data[col].round(2).tolist()}
        for col in data.columns
    ]
    return {"series": series, "y_axis_title": "Weight (%)"}


def summarize_style_run(run: Any) -> dict[str, object]:
    """Build the full chart + metrics summary for a StyleRun.

    Returns a dict with keys: chart_growth, chart_tracking,
    weights_history, weights_table, metrics, warnings.
    """
    chart_growth: Optional[dict[str, object]] = None
    chart_tracking: Optional[dict[str, object]] = None
    wh: Optional[dict[str, object]] = None
    weights_table: list[dict[str, object]] = []
    metrics: Optional[dict[str, object]] = None
    warnings: list[str] = []

    weights = run.weights
    if weights.empty:
        warnings.append("No feasible weights found for the requested configuration.")
    else:
        min_weight = float(weights.min().min())
        if min_weight < -1e-8:
            warnings.append("Some weights are negative beyond tolerance.")

        row_sum = weights.sum(axis=1)
        if ((row_sum - 1.0).abs() > 1e-5).any():
            warnings.append("Weights do not sum to 100% on every rebalance date.")

        wh = weights_history_payload(weights)

        top = _top_weights(weights.iloc[-1], top_n=10)
        if top.empty:
            warnings.append("No usable weights found for the latest rebalance date.")
        else:
            weights_table = [
                {
                    "name": str(name),
                    "weight": float(weight),
                    "weight_pct": float(weight * 100.0),
                }
                for name, weight in top.items()
            ]

    portfolio = run.portfolio_return.rename("Simulated Portfolio")
    benchmark = run.benchmark_return.rename("Market Benchmark")
    growth = pd.concat([portfolio, benchmark], axis=1).dropna(how="any")
    if not growth.empty:
        growth = np.exp(growth.cumsum()) * 100
        chart_growth = line_chart_payload(
            growth,
            y_axis_title="Indexed growth (base = 100)",
        )

    active_return = run.tracking_error.rename("Active Return").dropna()
    if not active_return.empty:
        active_return_pct = active_return.mul(100.0).rename("Residual return")
        chart_tracking = line_chart_payload(
            active_return_pct.to_frame(),
            y_axis_title="Daily return",
            round_to=4,
        )
        control_sd = float(active_return_pct.std())
        if np.isfinite(control_sd) and control_sd > 0:
            chart_tracking["control_limits"] = {
                "upper": float(3.0 * control_sd),
                "lower": float(-3.0 * control_sd),
            }

    te = run.tracking_error.dropna()
    window_frequency = str(run.params.get("window_frequency") or "daily")
    steps_per_year = STEPS_PER_YEAR.get(window_frequency, 252)

    info_ratio = None
    if not te.empty:
        te_vol = float(te.std())
        if np.isfinite(te_vol) and te_vol > 0:
            info_ratio = float((float(te.mean()) / te_vol) * np.sqrt(steps_per_year))

    sample = pd.concat([portfolio, benchmark], axis=1).dropna(how="any")
    sample_start = sample.index.min() if not sample.empty else None
    sample_end = sample.index.max() if not sample.empty else None
    sample_years = (
        float((sample_end - sample_start).days / 365.25)
        if sample_start is not None and sample_end is not None
        else None
    )

    perf_port = performance_summary(portfolio, steps_per_year=steps_per_year)
    perf_bench = performance_summary(benchmark, steps_per_year=steps_per_year)
    perf_active = performance_summary(te, steps_per_year=steps_per_year)

    metrics = {
        "inputs": {
            "window": run.params.get("style_window"),
            "window_years": run.params.get("style_window_years"),
            "window_frequency": window_frequency,
            "rebalance": run.params.get("optimize_frequency"),
            "method": run.params.get("method"),
            "assets": weights.shape[1] if not weights.empty else 0,
            "rebalance_start": (
                weights.index.min().strftime("%Y-%m-%d") if not weights.empty else None
            ),
            "rebalance_end": (
                weights.index.max().strftime("%Y-%m-%d") if not weights.empty else None
            ),
        },
        "sample": {
            "start": sample_start.strftime("%Y-%m-%d") if sample_start is not None else None,
            "end": sample_end.strftime("%Y-%m-%d") if sample_end is not None else None,
            "years": sample_years,
        },
        "performance": [
            {
                "name": "Simulated Portfolio",
                "total_cont_return": perf_port["total_return"],
                "ann_cont_return": perf_port["ann_return"],
                "ann_vol": perf_port["ann_vol"],
                "sharpe": perf_port["sharpe"],
                "info_ratio": None,
            },
            {
                "name": "Market Benchmark",
                "total_cont_return": perf_bench["total_return"],
                "ann_cont_return": perf_bench["ann_return"],
                "ann_vol": perf_bench["ann_vol"],
                "sharpe": perf_bench["sharpe"],
                "info_ratio": None,
            },
            {
                "name": "Active Return",
                "total_cont_return": perf_active["total_return"],
                "ann_cont_return": perf_active["ann_return"],
                "ann_vol": perf_active["ann_vol"],
                "sharpe": perf_active["sharpe"],
                "info_ratio": info_ratio,
            },
        ],
    }

    return {
        "chart_growth": chart_growth,
        "chart_tracking": chart_tracking,
        "weights_history": wh,
        "weights_table": weights_table,
        "metrics": metrics,
        "warnings": warnings,
    }


__all__ = [
    "line_chart_payload",
    "summarize_style_run",
    "weights_history_payload",
]
