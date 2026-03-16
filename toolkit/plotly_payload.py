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


def regime_probabilities_payload(
    probabilities: pd.DataFrame,
    *,
    regime_labels: Optional[dict[int, str]] = None,
) -> dict[str, object]:
    """Convert regime probability DataFrame to a stacked-area Plotly payload.

    Parameters
    ----------
    probabilities
        DataFrame with columns ``regime_0``, ``regime_1``, … and a
        DatetimeIndex.  Values in [0, 1].
    regime_labels
        Optional mapping from regime index to human-readable label.
    """
    if probabilities.empty:
        return {"series": [], "y_axis_title": "Probability (%)"}

    data = probabilities.copy().clip(lower=0.0).mul(100.0)
    dates = [dt.strftime("%Y-%m-%d") for dt in data.index]

    series = []
    for i, col in enumerate(data.columns):
        label = col
        if regime_labels and i in regime_labels:
            label = f"Regime {i} ({regime_labels[i]})"
        series.append({
            "name": label,
            "x": dates,
            "y": data[col].round(2).tolist(),
        })

    return {"series": series, "y_axis_title": "Probability (%)"}


def summarize_regime_run(
    run: Any,
    *,
    regime_labels: Optional[dict[int, str]] = None,
    raw_series: Optional[pd.Series] = None,
) -> dict[str, object]:
    """Build the full chart + metrics summary for a RegimeRun.

    Returns a dict with keys: chart_probabilities, chart_series,
    regime_assignments, transition_matrix, regime_table, metrics,
    regression_stats, warnings.

    Parameters
    ----------
    raw_series
        The untransformed series. If provided and different from the
        transformed series, a dual-axis chart payload is produced so the
        frontend can show raw on the left axis and transformed on the right.
    """
    warnings: list[str] = []
    series_name = run.params.get("series_name", "series")

    # Stacked regime probabilities chart
    chart_probabilities = regime_probabilities_payload(
        run.smoothed_probabilities, regime_labels=regime_labels,
    )
    # Attach series_name as chart title
    chart_probabilities["title"] = series_name

    # Transformed series as line chart
    transformed_s = run.series
    series_df = transformed_s.to_frame(name=series_name)
    chart_series = line_chart_payload(
        series_df, y_axis_title=series_name,
    )
    chart_series["title"] = series_name

    # Dual-axis payload: raw (left) + transformed (right) if applicable
    chart_dual_axis = None
    if raw_series is not None and not raw_series.equals(transformed_s):
        raw_dates = [dt.strftime("%Y-%m-%d") for dt in raw_series.index]
        trans_dates = [dt.strftime("%Y-%m-%d") for dt in transformed_s.index]
        chart_dual_axis = {
            "title": series_name,
            "raw": {
                "name": raw_series.name or "Raw",
                "x": raw_dates,
                "y": raw_series.round(6).tolist(),
            },
            "transformed": {
                "name": f"{series_name} (transformed)",
                "x": trans_dates,
                "y": transformed_s.round(6).tolist(),
            },
        }

    # Regime assignments for overlay colouring
    assignments = run.regime_assignments
    regime_assignments_payload = {
        "x": [dt.strftime("%Y-%m-%d") for dt in assignments.index],
        "regimes": assignments.tolist(),
    }

    # Transition matrix
    tm = run.transition_matrix
    transition_matrix_payload = {
        "labels": list(tm.columns),
        "values": tm.round(4).values.tolist(),
    }

    # Regime summary table
    rp = run.regime_params
    ed = run.expected_durations
    regime_table = []
    for i in sorted(rp.keys()):
        label = ""
        if regime_labels and i in regime_labels:
            label = regime_labels[i]
        regime_table.append({
            "regime": i,
            "label": label,
            "mean": round(rp[i].get("mean", float("nan")), 4),
            "variance": round(rp[i].get("variance", float("nan")), 4),
            "expected_duration": round(float(ed.iloc[i]), 1) if i < len(ed) else None,
        })

    # Regression statistics (parameter estimates with significance)
    regression_stats = []
    model_params = run.results.get("model_params")
    pvalues = run.results.get("pvalues")
    bse = run.results.get("bse")
    tvalues = run.results.get("tvalues")
    if model_params is not None and len(model_params) > 0:
        for param_name in model_params.index:
            coef = float(model_params[param_name])
            entry: dict[str, Any] = {"name": param_name, "coef": round(coef, 6)}
            if bse is not None and param_name in bse.index:
                entry["std_err"] = round(float(bse[param_name]), 6)
            if tvalues is not None and param_name in tvalues.index:
                entry["t_stat"] = round(float(tvalues[param_name]), 4)
            if pvalues is not None and param_name in pvalues.index:
                pv = float(pvalues[param_name])
                entry["pvalue"] = round(pv, 6)
                entry["significant"] = pv < 0.05
            regression_stats.append(entry)

    # Model-fit metrics
    meta = run.meta
    metrics: dict[str, Any] = {
        "series_name": series_name,
        "k_regimes": run.params.get("k_regimes"),
        "n_obs": meta.get("n_obs"),
        "n_train": meta.get("n_train"),
        "aic": round(run.results.get("aic", float("nan")), 2),
        "bic": round(run.results.get("bic", float("nan")), 2),
        "log_likelihood": round(run.results.get("log_likelihood", float("nan")), 2),
        "train_end": run.params.get("train_end"),
        "start": meta.get("start_date"),
        "end": meta.get("end_date"),
        "converged": meta.get("converged"),
    }

    if not meta.get("converged"):
        warnings.append("Model did not fully converge. Results may be unreliable.")

    # Train-end vertical line shape for the frontend
    train_end_shape = None
    if run.params.get("train_end"):
        train_end_shape = {
            "date": run.params["train_end"],
            "label": "Training cutoff",
        }

    return {
        "chart_probabilities": chart_probabilities,
        "chart_series": chart_series,
        "chart_dual_axis": chart_dual_axis,
        "regime_assignments": regime_assignments_payload,
        "transition_matrix": transition_matrix_payload,
        "regime_table": regime_table,
        "regression_stats": regression_stats,
        "metrics": metrics,
        "train_end_marker": train_end_shape,
        "warnings": warnings,
    }


def summarize_regime_collection(
    collection: Any,
    *,
    regime_labels_map: Optional[dict[str, dict[int, str]]] = None,
    raw_series_map: Optional[dict[str, pd.Series]] = None,
) -> dict[str, object]:
    """Build chart payloads for a RegimeCollection.

    Parameters
    ----------
    collection
        A :class:`RegimeCollection` instance.
    regime_labels_map
        Optional mapping from regime name to per-regime labels.
    raw_series_map
        Optional mapping from regime name to untransformed raw series.
        When provided, individual summaries include dual-axis chart data.

    Returns
    -------
    dict
        Keys: ``individual_summaries``, ``collection_info``.
    """
    if not collection:
        return {
            "individual_summaries": {},
            "collection_info": {"n_regimes": 0, "regime_names": []},
        }

    regime_labels_map = regime_labels_map or {}
    raw_series_map = raw_series_map or {}

    # Per-regime summaries
    individual_summaries = {}
    for cfg, run in collection.entries:
        labels = regime_labels_map.get(cfg.name) or (
            dict(cfg.regime_labels) if cfg.regime_labels else None
        )
        raw_s = raw_series_map.get(cfg.name)
        individual_summaries[cfg.name] = summarize_regime_run(
            run, regime_labels=labels, raw_series=raw_s,
        )

    return {
        "individual_summaries": individual_summaries,
        "collection_info": {
            "n_regimes": len(collection),
            "regime_names": collection.names,
        },
    }


__all__ = [
    "line_chart_payload",
    "regime_probabilities_payload",
    "summarize_regime_collection",
    "summarize_regime_run",
    "summarize_style_run",
    "weights_history_payload",
]
