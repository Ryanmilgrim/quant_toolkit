"""Markov Switching regime detection for econometric feature engineering.

Fits Hidden Markov / Markov-Switching models (via statsmodels) to economic
time series and exposes the fitted regime probabilities as ML-ready features.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitching

from .transforms import TransformConfig, TransformType


# ---------------------------------------------------------------------------
# RegimeConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeConfig:
    """User-defined configuration for a single regime detection run.

    Bundles the FRED series identifier, transformation, model parameters,
    and metadata so that regime definitions are portable and saveable.
    """

    name: str
    fred_series_id: str
    transform: TransformConfig = field(default_factory=TransformConfig)
    description: str = ""
    k_regimes: int = 2
    switching_variance: bool = True
    switching_trend: bool = True
    model_type: str = "regression"
    train_end: str | None = None
    regime_labels: dict[int, str] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probabilities_to_dataframe(
    probs: Any,
    index: pd.Index,
    k_regimes: int,
) -> pd.DataFrame:
    """Convert a T x k_regimes array (or DataFrame) to a labelled DataFrame."""
    cols = [f"regime_{i}" for i in range(k_regimes)]
    # statsmodels returns a DataFrame — extract raw values to avoid column
    # alignment issues when assigning new column names.
    data = probs.values if isinstance(probs, pd.DataFrame) else np.asarray(probs)
    return pd.DataFrame(data, index=index, columns=cols)


def _extract_transition_matrix(
    fitted: Any,
    k_regimes: int,
) -> pd.DataFrame:
    """Build the k x k transition probability matrix from a fitted model."""
    mat = np.array(fitted.regime_transition)
    # regime_transition may be 3-D (1 x k x k) or 2-D (k x k);
    # squeeze to ensure we have exactly 2-D.
    mat = mat.squeeze()
    # statsmodels stores the transition matrix as k x k where mat[i, j] is
    # the probability of transitioning FROM regime j TO regime i.
    # We transpose so that mat[i, j] is FROM i TO j (row-stochastic).
    mat = mat.T
    labels = [f"regime_{i}" for i in range(k_regimes)]
    return pd.DataFrame(mat, index=labels, columns=labels)


def _extract_regime_params(
    fitted: Any,
    k_regimes: int,
) -> dict[int, dict[str, float]]:
    """Extract per-regime parameters (mean, variance) from a fitted model."""
    params: dict[int, dict[str, float]] = {}
    for i in range(k_regimes):
        regime: dict[str, float] = {}
        # Mean / intercept
        try:
            regime["mean"] = float(fitted.params[f"const[{i}]"])
        except (KeyError, IndexError):
            # May not have switching intercept
            try:
                regime["mean"] = float(fitted.params["const"])
            except (KeyError, IndexError):
                regime["mean"] = np.nan
        # Variance / sigma
        try:
            regime["variance"] = float(fitted.params[f"sigma2[{i}]"])
        except (KeyError, IndexError):
            try:
                regime["variance"] = float(fitted.params["sigma2"])
            except (KeyError, IndexError):
                regime["variance"] = np.nan
        params[i] = regime
    return params


def _reorder_regimes(
    smoothed: pd.DataFrame,
    filtered: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    regime_params: dict[int, dict[str, float]],
    k_regimes: int,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, dict[str, float]]
]:
    """Reorder regimes so that regime 0 has the lowest mean.

    This ensures semantic stability across refits (e.g. regime 0 is always
    'recession' for GDP growth).
    """
    means = [regime_params[i].get("mean", np.nan) for i in range(k_regimes)]
    order = np.argsort(means)
    if np.array_equal(order, np.arange(k_regimes)):
        return smoothed, filtered, transition_matrix, regime_params

    # Reorder probability columns
    new_cols = [f"regime_{i}" for i in range(k_regimes)]
    old_cols = [f"regime_{o}" for o in order]
    smoothed_new = smoothed[old_cols].copy()
    smoothed_new.columns = new_cols
    filtered_new = filtered[old_cols].copy()
    filtered_new.columns = new_cols

    # Reorder transition matrix rows and columns
    tm = transition_matrix.to_numpy()
    tm_reordered = tm[np.ix_(order, order)]
    transition_matrix_new = pd.DataFrame(
        tm_reordered, index=new_cols, columns=new_cols,
    )

    # Reorder regime_params
    regime_params_new = {i: regime_params[int(order[i])] for i in range(k_regimes)}

    return smoothed_new, filtered_new, transition_matrix_new, regime_params_new


# ---------------------------------------------------------------------------
# RegimeRun
# ---------------------------------------------------------------------------

@dataclass
class RegimeRun:
    """Container for one regime-detection run.

    Mirrors ``FactorRun`` / ``StyleRun``: ``params`` holds configuration,
    ``results`` holds all computed outputs.
    """

    params: dict[str, Any]
    results: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.results[key]

    # --- Convenience properties ---

    @property
    def smoothed_probabilities(self) -> pd.DataFrame:
        return self.results["smoothed_probabilities"]

    @property
    def filtered_probabilities(self) -> pd.DataFrame:
        return self.results["filtered_probabilities"]

    @property
    def regime_assignments(self) -> pd.Series:
        return self.results["regime_assignments"]

    @property
    def transition_matrix(self) -> pd.DataFrame:
        return self.results["transition_matrix"]

    @property
    def regime_params(self) -> dict[int, dict[str, float]]:
        return self.results["regime_params"]

    @property
    def series(self) -> pd.Series:
        return self.results["series"]

    @property
    def expected_durations(self) -> pd.Series:
        return self.results["expected_durations"]

    @property
    def meta(self) -> dict[str, Any]:
        return self.results.get("meta", {})

    # --- Feature export ---

    def features(self, *, prefix: str | None = None) -> pd.DataFrame:
        """Return an ML-ready DataFrame of regime features.

        Columns
        -------
        {prefix}_prob_0, {prefix}_prob_1, ... : smoothed regime probabilities
        {prefix}_regime : integer regime assignment (argmax of smoothed probs)
        """
        pfx = prefix or self.params.get("series_name", "regime")
        probs = self.smoothed_probabilities.copy()
        probs.columns = [f"{pfx}_prob_{i}" for i in range(probs.shape[1])]
        assign = self.regime_assignments.rename(f"{pfx}_regime")
        return pd.concat([probs, assign], axis=1)

    # --- Display ---

    def summary(self) -> str:
        """Human-readable summary of the regime-detection run."""
        p = self.params
        m = self.meta
        lines = [
            "RegimeRun",
            f"  series:             {p.get('series_name', '?')}",
            f"  k_regimes:          {p.get('k_regimes')}",
            f"  switching_variance: {p.get('switching_variance')}",
            f"  switching_trend:    {p.get('switching_trend')}",
            f"  model_type:         {p.get('model_type')}",
            f"  n_obs:              {m.get('n_obs', '?')}",
            f"  train_end:          {p.get('train_end', 'full sample')}",
            f"  converged:          {m.get('converged', '?')}",
            f"  AIC:                {self.results.get('aic', '?'):.4f}"
            if isinstance(self.results.get("aic"), (int, float))
            else f"  AIC:                {self.results.get('aic', '?')}",
            f"  BIC:                {self.results.get('bic', '?'):.4f}"
            if isinstance(self.results.get("bic"), (int, float))
            else f"  BIC:                {self.results.get('bic', '?')}",
            "",
            "  Regime Parameters:",
        ]
        rp = self.regime_params
        ed = self.expected_durations
        for i in sorted(rp.keys()):
            mean = rp[i].get("mean", np.nan)
            var = rp[i].get("variance", np.nan)
            dur = ed.iloc[i] if i < len(ed) else np.nan
            lines.append(
                f"    regime {i}: mean={mean:+.4f}, var={var:.4f}, "
                f"E[duration]={dur:.1f} periods"
            )
        lines.append("")
        lines.append("  Transition Matrix:")
        tm = self.transition_matrix
        for row_label in tm.index:
            row_vals = "  ".join(f"{v:.4f}" for v in tm.loc[row_label])
            lines.append(f"    {row_label}: {row_vals}")
        return "\n".join(lines)

    # --- Plotting (delegate to regime_charts) ---

    def plot_regime_probabilities(self, *, ax=None, **kwargs):
        from toolkit.regime_charts import plot_regime_probabilities

        return plot_regime_probabilities(
            self.smoothed_probabilities, ax=ax, **kwargs,
        )

    def plot_regime_series(self, *, ax=None, **kwargs):
        from toolkit.regime_charts import plot_regime_series

        return plot_regime_series(
            self.series,
            self.regime_assignments,
            k_regimes=self.params["k_regimes"],
            ax=ax,
            **kwargs,
        )

    def plot_transition_matrix(self, *, ax=None, **kwargs):
        from toolkit.regime_charts import plot_transition_matrix

        return plot_transition_matrix(
            self.transition_matrix, ax=ax, **kwargs,
        )


# ---------------------------------------------------------------------------
# RegimeModel
# ---------------------------------------------------------------------------

class RegimeModel:
    """Fit a Markov Switching model to an economic time series.

    Parameters
    ----------
    k_regimes
        Number of hidden regimes (2-5).
    switching_variance
        Whether the variance switches across regimes.
    switching_trend
        Whether the intercept/trend switches across regimes.
    model_type
        ``"regression"`` uses ``MarkovRegression`` (allows exogenous vars).
        ``"switching"`` uses ``MarkovSwitching``.
    switching_exog
        Whether exogenous coefficients switch (only relevant with exog data).
    """

    def __init__(
        self,
        *,
        k_regimes: int = 2,
        switching_variance: bool = True,
        switching_trend: bool = True,
        model_type: str = "regression",
        switching_exog: bool = False,
    ) -> None:
        if k_regimes < 2 or k_regimes > 5:
            raise ValueError("k_regimes must be between 2 and 5")
        if model_type not in ("regression", "switching"):
            raise ValueError("model_type must be 'regression' or 'switching'")
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        self.switching_trend = switching_trend
        self.model_type = model_type
        self.switching_exog = switching_exog

    @classmethod
    def from_config(cls, config: RegimeConfig) -> "RegimeModel":
        """Create a RegimeModel from a :class:`RegimeConfig`."""
        return cls(
            k_regimes=config.k_regimes,
            switching_variance=config.switching_variance,
            switching_trend=config.switching_trend,
            model_type=config.model_type,
        )

    def run(
        self,
        series: pd.Series,
        *,
        exog: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
        train_end: str | pd.Timestamp | None = None,
        search_reps: int = 25,
    ) -> RegimeRun:
        """Fit the Markov Switching model and return a :class:`RegimeRun`.

        Parameters
        ----------
        series
            Univariate time series with a DatetimeIndex.
        exog
            Optional exogenous variables (DataFrame aligned with *series*).
        name
            Descriptive name for the series (defaults to ``series.name``).
        train_end
            If provided, fit the model only on data up to this date, then
            apply the fitted parameters to the full series to obtain
            out-of-sample regime probabilities.
        search_reps
            Number of random starting-value searches for convergence.
        """
        # --- Validate input ---
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pd.Series")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("series must have a DatetimeIndex")
        s = series.dropna()
        if len(s) < 20:
            raise ValueError("series must have at least 20 non-NaN observations")

        series_name = name or series.name or "series"

        # --- Split for train_end ---
        train_end_ts: pd.Timestamp | None = None
        if train_end is not None:
            train_end_ts = pd.Timestamp(train_end)
            train_s = s.loc[:train_end_ts]
            if len(train_s) < 20:
                raise ValueError(
                    f"Training set (up to {train_end}) has only "
                    f"{len(train_s)} observations; need at least 20"
                )
        else:
            train_s = s

        # --- Build and fit on training data ---
        fitted, converged = self._fit_model(
            train_s, exog=exog, search_reps=search_reps, series_name=series_name,
        )

        # --- Get full-sample probabilities ---
        if train_end_ts is not None and len(s) > len(train_s):
            # Apply fitted params to full series via smooth()
            full_smoothed, full_filtered = self._apply_params_to_full(
                s, fitted, exog=exog,
            )
        else:
            full_smoothed = _probabilities_to_dataframe(
                fitted.smoothed_marginal_probabilities, s.index, self.k_regimes,
            )
            full_filtered = _probabilities_to_dataframe(
                fitted.filtered_marginal_probabilities, s.index, self.k_regimes,
            )

        # --- Extract results ---
        transition_matrix = _extract_transition_matrix(fitted, self.k_regimes)
        regime_params = _extract_regime_params(fitted, self.k_regimes)

        # Reorder regimes so regime 0 = lowest mean
        full_smoothed, full_filtered, transition_matrix, regime_params = (
            _reorder_regimes(
                full_smoothed, full_filtered, transition_matrix,
                regime_params, self.k_regimes,
            )
        )

        regime_assignments = pd.Series(
            full_smoothed.values.argmax(axis=1),
            index=s.index,
            name="regime",
            dtype=int,
        )

        # Expected durations: 1 / (1 - p_ii)
        diag = np.diag(transition_matrix.to_numpy())
        expected_durations = pd.Series(
            1.0 / np.clip(1.0 - diag, 1e-12, None),
            index=[f"regime_{i}" for i in range(self.k_regimes)],
            name="expected_duration",
        )

        model_params = pd.Series(fitted.params, name="model_params")

        # Regression statistics (pvalues, std errors, t-values)
        try:
            pvalues = pd.Series(fitted.pvalues, name="pvalues")
        except Exception:
            pvalues = pd.Series(dtype=float, name="pvalues")
        try:
            bse = pd.Series(fitted.bse, name="bse")
        except Exception:
            bse = pd.Series(dtype=float, name="bse")
        try:
            tvalues = pd.Series(fitted.tvalues, name="tvalues")
        except Exception:
            tvalues = pd.Series(dtype=float, name="tvalues")

        meta: dict[str, Any] = {
            "fit_date": datetime.now().isoformat(),
            "n_obs": len(s),
            "n_train": len(train_s),
            "converged": converged,
            "start_date": s.index.min().date().isoformat(),
            "end_date": s.index.max().date().isoformat(),
        }
        if train_end_ts is not None:
            meta["train_end"] = train_end_ts.date().isoformat()

        results: dict[str, Any] = {
            "smoothed_probabilities": full_smoothed,
            "filtered_probabilities": full_filtered,
            "regime_assignments": regime_assignments,
            "transition_matrix": transition_matrix,
            "regime_params": regime_params,
            "series": s,
            "expected_durations": expected_durations,
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "log_likelihood": float(fitted.llf),
            "model_params": model_params,
            "pvalues": pvalues,
            "bse": bse,
            "tvalues": tvalues,
            "meta": meta,
        }

        params: dict[str, Any] = {
            "series_name": series_name,
            "k_regimes": self.k_regimes,
            "switching_variance": self.switching_variance,
            "switching_trend": self.switching_trend,
            "model_type": self.model_type,
            "switching_exog": self.switching_exog,
            "train_end": train_end_ts.date().isoformat() if train_end_ts else None,
        }

        return RegimeRun(params=params, results=results)

    def _fit_model(
        self,
        s: pd.Series,
        *,
        exog: Optional[pd.DataFrame],
        search_reps: int,
        series_name: str,
    ) -> tuple[Any, bool]:
        """Build and fit the statsmodels model with convergence retries."""
        if self.model_type == "regression":
            model = MarkovRegression(
                s,
                k_regimes=self.k_regimes,
                trend="c",
                switching_variance=self.switching_variance,
                switching_trend=self.switching_trend,
                exog=exog,
                switching_exog=self.switching_exog,
            )
        else:
            model = MarkovSwitching(
                s,
                k_regimes=self.k_regimes,
                switching_variance=self.switching_variance,
            )

        converged = False
        try:
            fitted = model.fit(search_reps=search_reps, disp=False)
            converged = True
        except Exception:
            try:
                fitted = model.fit(search_reps=search_reps * 4, disp=False)
                converged = True
            except Exception as exc:
                raise RuntimeError(
                    f"Markov Switching model failed to converge for "
                    f"'{series_name}' after retries: {exc}"
                ) from exc
        return fitted, converged

    def _apply_params_to_full(
        self,
        full_series: pd.Series,
        fitted: Any,
        *,
        exog: Optional[pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply trained parameters to the full series via smooth()."""
        if self.model_type == "regression":
            full_model = MarkovRegression(
                full_series,
                k_regimes=self.k_regimes,
                trend="c",
                switching_variance=self.switching_variance,
                switching_trend=self.switching_trend,
                exog=exog,
                switching_exog=self.switching_exog,
            )
        else:
            full_model = MarkovSwitching(
                full_series,
                k_regimes=self.k_regimes,
                switching_variance=self.switching_variance,
            )

        full_result = full_model.smooth(fitted.params)
        smoothed = _probabilities_to_dataframe(
            full_result.smoothed_marginal_probabilities,
            full_series.index, self.k_regimes,
        )
        filtered = _probabilities_to_dataframe(
            full_result.filtered_marginal_probabilities,
            full_series.index, self.k_regimes,
        )
        return smoothed, filtered


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------

def _fit_single(
    name: str,
    series: pd.Series,
    config: dict[str, Any],
) -> tuple[str, RegimeRun]:
    """Fit a single series. Suitable for parallel dispatch."""
    train_end = config.pop("train_end", None)
    model = RegimeModel(**config)
    run = model.run(series, name=name, train_end=train_end)
    return name, run


def fit_regime_batch(
    series_dict: dict[str, pd.Series],
    *,
    configs: Optional[dict[str, dict[str, Any]]] = None,
    default_config: Optional[dict[str, Any]] = None,
) -> dict[str, RegimeRun]:
    """Fit regime models to multiple series in parallel.

    Parameters
    ----------
    series_dict
        Mapping of series name to pd.Series.
    configs
        Optional per-series config overrides (keys matching *series_dict*).
    default_config
        Default RegimeModel kwargs for series without a per-series config.

    Returns
    -------
    dict[str, RegimeRun]
    """
    if default_config is None:
        default_config = {}
    if configs is None:
        configs = {}

    args_list = []
    for name, series in series_dict.items():
        cfg = configs.get(name, default_config).copy()
        args_list.append((name, series, cfg))

    max_workers = min(len(args_list), os.cpu_count() or 4)
    if max_workers <= 1 or len(args_list) <= 1:
        results_list = [_fit_single(*a) for a in args_list]
    else:
        results_list = [None] * len(args_list)  # type: ignore[list-item]
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_idx = {
                    pool.submit(_fit_single, *a): i
                    for i, a in enumerate(args_list)
                }
                for fut in as_completed(future_to_idx):
                    results_list[future_to_idx[fut]] = fut.result()
        except Exception:
            results_list = [_fit_single(*a) for a in args_list]

    return {name: run for name, run in results_list}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RegimeCollection
# ---------------------------------------------------------------------------

class RegimeCollection:
    """Iteratively accumulated collection of named regime detection runs.

    Each entry pairs a :class:`RegimeConfig` with its :class:`RegimeRun`.
    The collection's main output is :meth:`endogenous_data` — a combined
    multivariate DataFrame of soft regime probabilities suitable for
    downstream ML models.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[RegimeConfig, RegimeRun]] = []

    @property
    def entries(self) -> list[tuple[RegimeConfig, RegimeRun]]:
        return list(self._entries)

    @property
    def names(self) -> list[str]:
        return [cfg.name for cfg, _ in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def add(self, config: RegimeConfig, run: RegimeRun) -> None:
        """Add a regime to the collection.

        Raises ``ValueError`` if a regime with the same name already exists.
        """
        if config.name in self.names:
            raise ValueError(
                f"A regime named '{config.name}' already exists in the "
                f"collection. Remove it first or use a different name."
            )
        self._entries.append((config, run))

    def remove(self, name: str) -> None:
        """Remove a regime by name."""
        idx = next(
            (i for i, (cfg, _) in enumerate(self._entries) if cfg.name == name),
            None,
        )
        if idx is None:
            raise KeyError(f"No regime named '{name}' in collection")
        del self._entries[idx]

    def get(self, name: str) -> tuple[RegimeConfig, RegimeRun]:
        """Get a (config, run) pair by regime name."""
        for cfg, run in self._entries:
            if cfg.name == name:
                return cfg, run
        raise KeyError(f"No regime named '{name}' in collection")

    def endogenous_data(self) -> pd.DataFrame:
        """Build the combined multivariate soft-probability DataFrame.

        For each regime entry, the smoothed probability columns are included
        with a name-based prefix to avoid column collisions. The result is
        an outer-joined, forward-filled DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns like ``"{name}_prob_0"``, ``"{name}_prob_1"``, etc.
        """
        if not self._entries:
            return pd.DataFrame()

        frames = []
        for cfg, run in self._entries:
            probs = run.smoothed_probabilities.copy()
            prefix = cfg.name.replace(" ", "_")
            probs.columns = [
                f"{prefix}_prob_{i}" for i in range(probs.shape[1])
            ]
            frames.append(probs)

        combined = pd.concat(frames, axis=1, join="outer")
        combined = combined.ffill().dropna(how="all")
        return combined

    def summary(self) -> str:
        """Human-readable summary of all regimes in the collection."""
        if not self._entries:
            return "RegimeCollection (empty)"

        lines = [f"RegimeCollection ({len(self._entries)} regimes)"]
        lines.append("-" * 60)
        for cfg, run in self._entries:
            m = run.meta
            lines.append(
                f"  {cfg.name}: {cfg.fred_series_id} "
                f"({cfg.transform.description}) "
                f"k={cfg.k_regimes}, "
                f"n_obs={m.get('n_obs', '?')}, "
                f"train_end={cfg.train_end or 'full'}"
            )
        endo = self.endogenous_data()
        if not endo.empty:
            lines.append("")
            lines.append(f"  Endogenous data: {endo.shape[1]} columns, "
                         f"{endo.shape[0]} rows")
            lines.append(f"  Date range: {endo.index.min()} → {endo.index.max()}")
        return "\n".join(lines)


__all__ = [
    "RegimeCollection",
    "RegimeConfig",
    "RegimeModel",
    "RegimeRun",
    "fit_regime_batch",
]
