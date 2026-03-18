"""Risk model with PCA covariance and GARCH conditional variance.

Pipeline
--------
1. Per-asset factor regressions (LassoCV selection + post-OLS) for beta loadings
2. PCA (SVD) on factor returns to extract principal components
3. GARCH(1,1) (arch library; EWMA fallback) for residual and PC conditional variance
4. Reconstruct time-varying asset covariance from stored eigenvectors + variances

Data
----
Accepts either a MultiIndex ``uni`` DataFrame from
:func:`toolkit.universe.get_universe_returns` (with ``"assets"``,
``"factors"``, and ``"benchmarks"`` groups) **or** explicit ``assets``,
``factors``, and ``rf`` arguments.

Important outputs
-----------------
- ``beta_loadings``: per-asset factor betas (LassoCV + post-OLS)
- ``eigen_vectors``: PCA loadings mapping factors to principal components
- ``resid_cond_var`` / ``pc_cond_var``: one-step-ahead conditional variances
- ``asset_cov_forecast``: latest next-day asset covariance matrix
- ``evaluation``: optional train/test backtest of vol and correlation forecasts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def annualize_vol(
    vol: Union[pd.Series, pd.DataFrame],
    *,
    trading_days: int = 252,
) -> Union[pd.Series, pd.DataFrame]:
    """Annualize a daily volatility series using the square-root-of-time rule."""
    return vol * float(np.sqrt(int(trading_days)))


def covariance_to_correlation(
    cov: np.ndarray,
    *,
    eps: float = 1e-12,
    clip: bool = False,
) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Guards against divide-by-zero by zeroing correlations when vol ~ 0.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square (N x N) matrix")

    var = np.diag(cov)
    vol = np.sqrt(np.clip(var, 0.0, None))
    denom = np.outer(vol, vol)

    corr = np.zeros_like(cov, dtype=float)
    np.divide(cov, denom, out=corr, where=denom > eps)

    if clip:
        corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def covariance_to_correlation_df(
    cov_df: pd.DataFrame,
    *,
    eps: float = 1e-12,
    clip: bool = False,
) -> pd.DataFrame:
    """DataFrame wrapper for :func:`covariance_to_correlation`."""
    corr = covariance_to_correlation(cov_df.to_numpy(dtype=float), eps=eps, clip=clip)
    return pd.DataFrame(corr, index=cov_df.index, columns=cov_df.columns)


# ---------------------------------------------------------------------------
# Progress helper (tqdm wrapper)
# ---------------------------------------------------------------------------

def _iter_with_progress(
    iterable,
    *,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
):
    """Wrap *iterable* with tqdm if available, otherwise pass through."""
    if disable:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
        import sys
        return tqdm(iterable, desc=desc, unit=unit, disable=disable,
                    file=sys.stdout, dynamic_ncols=True)
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# sklearn version compatibility
# ---------------------------------------------------------------------------

def _sklearn_version_tuple() -> Tuple[int, int, int]:
    """Return scikit-learn version as (major, minor, patch)."""
    try:
        import sklearn  # type: ignore[import-untyped]
        v = getattr(sklearn, "__version__", "0.0.0")
    except Exception:
        v = "0.0.0"
    parts = str(v).split(".")
    nums: List[int] = []
    for p in parts[:3]:
        digits = "".join(ch for ch in p if ch.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------

def _fit_ols(y: pd.Series, X: pd.DataFrame):
    """Fit OLS with an intercept column named ``'alpha'``."""
    import statsmodels.api as sm  # type: ignore[import-untyped]

    y = pd.to_numeric(y, errors="coerce")
    X = X.apply(pd.to_numeric, errors="coerce")
    if y.name is None:
        y = y.rename("y")
    X = X.copy()
    X.insert(0, "alpha", 1.0)
    df = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    y2 = df[y.name]
    X2 = df.drop(columns=y.name)
    return sm.OLS(y2, X2).fit()


# ---------------------------------------------------------------------------
# Beta estimation (LassoCV + post-OLS + t-stat pruning)
# ---------------------------------------------------------------------------

def _rank_abs_corr(y: pd.Series, X: pd.DataFrame) -> List[str]:
    """Rank factors by absolute correlation with *y*."""
    yv = y.to_numpy(dtype=float)
    corrs: Dict[str, float] = {}
    for c in X.columns:
        xv = X[c].to_numpy(dtype=float)
        if len(yv) > 2 and np.std(yv) > 0 and np.std(xv) > 0:
            cc = float(np.corrcoef(yv, xv)[0, 1])
            corrs[c] = abs(cc) if np.isfinite(cc) else 0.0
        else:
            corrs[c] = 0.0
    return sorted(list(X.columns), key=lambda name: corrs.get(name, 0.0), reverse=True)


def _estimate_asset_betas(
    *,
    y: pd.Series,
    X: pd.DataFrame,
    max_exhaustive_factors: int,
    lasso_cv_splits: int,
    lasso_alphas: int,
    lasso_max_iter: int,
    lasso_coef_tol: float,
    tstat_cutoff: float,
    min_factors: int,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """Fit one asset regression via LassoCV selection + post-OLS refit.

    Returns ``(ols_result, selected_factors, meta_dict)``.
    """
    from sklearn.linear_model import LassoCV  # type: ignore[import-untyped]
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]
    import sklearn  # type: ignore[import-untyped]

    y = pd.to_numeric(y, errors="coerce")
    X = X.apply(pd.to_numeric, errors="coerce")
    if y.name is None:
        y = y.rename("y")

    df = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    y2 = df[y.name]
    X2 = df.drop(columns=y.name)

    meta: Dict[str, Any] = {}
    if X2.shape[1] == 0 or len(df) < 10:
        res = _fit_ols(y2, pd.DataFrame(index=y2.index))
        return res, [], meta

    # Drop near-constant factors
    var = X2.var(axis=0)
    X2 = X2.loc[:, var > 1e-18]
    if X2.shape[1] == 0:
        res = _fit_ols(y2, pd.DataFrame(index=y2.index))
        return res, [], meta

    # Cap factors by absolute correlation for stability
    ranked = _rank_abs_corr(y2, X2)
    if max_exhaustive_factors > 0 and X2.shape[1] > max_exhaustive_factors:
        keep = ranked[:max_exhaustive_factors]
        X2 = X2[keep]
        ranked = keep

    Xv = X2.to_numpy(dtype=float)
    mu = np.nanmean(Xv, axis=0)
    sd = np.nanstd(Xv, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xz = (Xv - mu) / sd

    n = len(df)
    splits = max(2, min(lasso_cv_splits, max(2, n // 10), n - 1)) if n > 3 else 2
    cv = TimeSeriesSplit(n_splits=splits)

    sk_ver = _sklearn_version_tuple()
    if sk_ver >= (1, 7, 0):
        lasso = LassoCV(alphas=int(lasso_alphas), cv=cv, fit_intercept=True,
                        max_iter=int(lasso_max_iter))
    else:
        lasso = LassoCV(n_alphas=int(lasso_alphas), cv=cv, fit_intercept=True,
                        max_iter=int(lasso_max_iter))

    lasso.fit(Xz, y2.to_numpy(dtype=float))
    coef = np.asarray(lasso.coef_).reshape(-1)

    sel = [c for c, v in zip(X2.columns, coef) if abs(float(v)) > lasso_coef_tol]
    if len(sel) < min_factors:
        sel = list(ranked[:min_factors])

    # Post-OLS on original scale
    res = _fit_ols(y2, X2[sel] if sel else pd.DataFrame(index=y2.index))
    sel_final = list(sel)

    # t-stat pruning
    cutoff = float(tstat_cutoff)
    while cutoff > 0 and sel_final and len(sel_final) > min_factors:
        tvals = pd.Series(res.tvalues, index=res.params.index).drop("alpha", errors="ignore")
        tabs = tvals.reindex(sel_final).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bad = tabs[tabs < cutoff]
        if bad.empty:
            break
        worst = str(bad.idxmin())
        sel_final = [f for f in sel_final if f != worst]
        res = _fit_ols(y2, X2[sel_final] if sel_final else pd.DataFrame(index=y2.index))

    meta = {
        "nobs": int(getattr(res, "nobs", len(df))),
        "cv_splits": int(splits),
        "lasso_alpha": float(getattr(lasso, "alpha_", np.nan)),
        "lasso_alphas_grid": np.asarray(getattr(lasso, "alphas_", np.array([])), dtype=float),
        "factors_in_lasso": list(X2.columns),
        "sklearn_version": str(getattr(sklearn, "__version__", "")),
    }
    return res, sel_final, meta


# ---------------------------------------------------------------------------
# PCA via SVD
# ---------------------------------------------------------------------------

def _fit_pca_svd(
    factors_train: pd.DataFrame,
    *,
    demean: bool,
    n_components: Optional[int],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Fit PCA on factor returns via SVD.

    Returns ``(eigen_vectors, singular_values)`` where ``eigen_vectors``
    is a DataFrame of shape ``[n_factors x n_pcs]``.
    """
    Xdf = factors_train.replace([np.inf, -np.inf], np.nan).dropna(how="any").astype(float)
    if Xdf.shape[0] < 3 or Xdf.shape[1] < 1:
        ev = pd.DataFrame(index=factors_train.columns,
                          data=np.zeros((factors_train.shape[1], 0)))
        return ev, pd.Series(dtype=float)

    X = Xdf.to_numpy(dtype=float)
    if demean:
        X = X - np.mean(X, axis=0, keepdims=True)

    _, singular_values, right_t = np.linalg.svd(X, full_matrices=False)
    eigen_vectors_full = right_t.T  # (n_factors x n_pcs)

    k_full = eigen_vectors_full.shape[1]
    k = k_full if n_components is None else int(max(1, min(k_full, int(n_components))))

    eigen_vectors = eigen_vectors_full[:, :k]
    svals = singular_values[:k]

    pc_cols = [f"PC{i + 1}" for i in range(k)]
    ev_df = pd.DataFrame(eigen_vectors, index=Xdf.columns, columns=pc_cols)
    sv_s = pd.Series(svals, index=pc_cols, name="singular_value")
    return ev_df, sv_s


def _transform_pca(
    factors: pd.DataFrame,
    eigen_vectors: pd.DataFrame,
    *,
    demean: bool,
) -> pd.DataFrame:
    """Project factor returns into PC space using stored eigenvectors."""
    Xdf = factors.replace([np.inf, -np.inf], np.nan).astype(float)
    Xdf = Xdf.reindex(columns=eigen_vectors.index)
    X = Xdf.to_numpy(dtype=float)
    if demean:
        X = X - np.nanmean(X, axis=0, keepdims=True)
    Z = X @ eigen_vectors.to_numpy(dtype=float)
    return pd.DataFrame(Z, index=Xdf.index, columns=eigen_vectors.columns)


# ---------------------------------------------------------------------------
# GARCH(1,1) fitting and filtering
# ---------------------------------------------------------------------------

def _safe_get(params: pd.Series, key: str, default: float = np.nan) -> float:
    try:
        return float(params.loc[key])
    except Exception:
        return float(default)


def _fit_garch(
    y_train: pd.Series,
    *,
    mean: str,
    dist: str,
    scale: float,
    min_obs: int,
    ewma_lambda: float,
) -> Dict[str, Any]:
    """Fit GARCH(1,1) on *y_train* and return parameter dict.

    Falls back to EWMA if the arch library fails or there are too few observations.
    """
    y = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(y))

    if n < min_obs:
        return _ewma_params(y, mean=mean, ewma_lambda=ewma_lambda,
                            reason="too_few_obs", nobs=n)

    try:
        from arch import arch_model  # type: ignore[import-untyped]
    except ImportError:
        return _ewma_params(y, mean=mean, ewma_lambda=ewma_lambda,
                            reason="arch_not_installed", nobs=n)

    ys = (y * scale).astype(float)
    try:
        am = arch_model(ys, mean=mean, vol="GARCH", p=1, o=0, q=1,
                        dist=dist, rescale=False)
        res = am.fit(update_freq=0, disp="off", show_warning=False)
        params = getattr(res, "params", pd.Series(dtype=float))

        mu_s = _safe_get(params, "mu", 0.0)
        omega_s = _safe_get(params, "omega", np.nan)
        alpha = _safe_get(params, "alpha[1]", np.nan)
        beta = _safe_get(params, "beta[1]", np.nan)

        mu = float(mu_s) / scale
        omega = float(omega_s) / (scale * scale) if np.isfinite(omega_s) else np.nan

        if not (np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(omega)):
            raise ValueError("Non-finite GARCH parameters.")

        return {
            "ok": True,
            "method": "arch_garch11",
            "mu": float(mu),
            "omega": float(max(omega, 0.0)),
            "alpha": float(max(alpha, 0.0)),
            "beta": float(max(beta, 0.0)),
            "nobs": int(getattr(res, "nobs", n)),
            "mean": mean,
            "dist": dist,
            "scale": scale,
            "aic": float(getattr(res, "aic", np.nan)),
            "bic": float(getattr(res, "bic", np.nan)),
            "loglikelihood": float(getattr(res, "loglikelihood", np.nan)),
        }
    except Exception as e:
        return _ewma_params(y, mean=mean, ewma_lambda=ewma_lambda,
                            reason="arch_fit_failed", nobs=n, error=str(e))


def _ewma_params(
    y: pd.Series,
    *,
    mean: str,
    ewma_lambda: float,
    reason: str,
    nobs: int,
    error: str = "",
) -> Dict[str, Any]:
    """Return an EWMA fallback parameter dict."""
    mu = float(y.mean()) if mean.lower() == "constant" else 0.0
    return {
        "ok": False,
        "method": "ewma_fallback",
        "reason": reason,
        "error": error,
        "mu": mu,
        "omega": np.nan,
        "alpha": np.nan,
        "beta": np.nan,
        "nobs": nobs,
        "mean": mean,
        "ewma_lambda": ewma_lambda,
    }


def _filter_garch(
    y_full: pd.Series,
    garch_params: Dict[str, Any],
    *,
    full_index: pd.Index,
) -> Dict[str, Any]:
    """Forward-filter conditional variance from GARCH parameters.

    Returns ``cond_var``, ``forecast_var``, and ``std_resid`` series.
    """
    y = pd.to_numeric(y_full, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = y.reindex(full_index)

    if not garch_params.get("ok", False):
        return _filter_ewma(y, garch_params, full_index=full_index)

    mu = float(garch_params["mu"])
    omega = float(garch_params["omega"])
    alpha = float(garch_params["alpha"])
    beta = float(garch_params["beta"])

    yv = y.to_numpy(dtype=float)
    e = yv - mu

    y_nonan = y.dropna()
    var0 = float(y_nonan.var(ddof=1)) if len(y_nonan) > 1 else 0.0
    ab = alpha + beta
    if np.isfinite(omega) and ab < 0.999 and omega > 0:
        var0 = float(max(var0, omega / max(1e-12, (1.0 - ab))))

    T = len(full_index)
    h = np.empty(T, dtype=float)
    fwd = np.empty(T, dtype=float)

    e_valid = np.isfinite(e)
    e2 = np.where(e_valid, e * e, 0.0)

    h[0] = max(var0, 0.0)
    for t in range(1, T):
        if e_valid[t - 1]:
            h[t] = omega + alpha * e2[t - 1] + beta * h[t - 1]
        else:
            h[t] = h[t - 1]
        if h[t] < 0.0:
            h[t] = 0.0

    # Forward variance — fully vectorized
    fwd = np.where(e_valid, omega + alpha * e2 + beta * h, h)
    np.maximum(fwd, 0.0, out=fwd)

    cond_vol = np.sqrt(np.clip(h, 0.0, None))
    std_resid = np.full(T, np.nan, dtype=float)
    mask = np.isfinite(yv) & np.isfinite(cond_vol) & (cond_vol > 0)
    std_resid[mask] = (e[mask] / cond_vol[mask])

    return {
        "cond_var": pd.Series(h, index=full_index, name="cond_var"),
        "forecast_var": pd.Series(fwd, index=full_index, name="forecast_var"),
        "std_resid": pd.Series(std_resid, index=full_index, name="std_resid"),
        "meta": dict(garch_params),
    }


def _filter_ewma(
    y: pd.Series,
    garch_params: Dict[str, Any],
    *,
    full_index: pd.Index,
) -> Dict[str, Any]:
    """EWMA fallback filter when GARCH fitting failed."""
    lam = float(garch_params.get("ewma_lambda", 0.94))
    mu = float(garch_params.get("mu", 0.0))

    yv = y.to_numpy(dtype=float)
    e = yv - mu

    T = len(full_index)
    h = np.empty(T, dtype=float)

    y_nonan = y.dropna()
    h0 = max(float(y_nonan.var(ddof=1)) if len(y_nonan) > 1 else 0.0, 0.0)
    one_minus_lam = 1.0 - lam

    e_valid = np.isfinite(e)
    e2 = np.where(e_valid, e * e, 0.0)

    h[0] = h0
    for t in range(1, T):
        if e_valid[t - 1]:
            h[t] = lam * h[t - 1] + one_minus_lam * e2[t - 1]
        else:
            h[t] = h[t - 1]

    # Forward variance — fully vectorized
    fwd = np.where(e_valid, lam * h + one_minus_lam * e2, h)

    cond_vol = np.sqrt(np.clip(h, 0.0, None))
    std = np.full(T, np.nan, dtype=float)
    mask = np.isfinite(yv) & np.isfinite(cond_vol) & (cond_vol > 0)
    std[mask] = e[mask] / cond_vol[mask]

    meta = dict(garch_params)
    meta.update({"ewma_lambda": lam, "mu": mu})
    return {
        "cond_var": pd.Series(h, index=full_index, name="cond_var"),
        "forecast_var": pd.Series(fwd, index=full_index, name="forecast_var"),
        "std_resid": pd.Series(std, index=full_index, name="std_resid"),
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# RiskModelRun dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskModelRun:
    """Container for one risk model run.

    Mirrors the ``StyleRun`` pattern: ``params`` holds configuration,
    ``results`` holds all computed outputs.  Access results via dict-style
    ``run["key"]`` or convenience properties.
    """

    params: dict[str, Any]
    results: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.results[key]

    # --- Convenience properties ---

    @property
    def meta(self) -> dict[str, Any]:
        return self.results.get("meta", {})

    @property
    def beta_loadings(self) -> pd.DataFrame:
        return self.results["beta_loadings"]

    @property
    def eigen_vectors(self) -> pd.DataFrame:
        """PCA eigenvectors (factor -> PC), shape ``[n_factors x n_pcs]``."""
        return self.results["eigen_vectors"]

    @property
    def resid_cond_var(self) -> pd.DataFrame:
        return self.results["resid_cond_var"]

    @property
    def pc_cond_var(self) -> pd.DataFrame:
        return self.results["pc_cond_var"]

    @property
    def assets(self) -> List[str]:
        return list(self.beta_loadings.index)

    @property
    def factors(self) -> List[str]:
        return list(self.beta_loadings.columns)

    # --- Time-varying beta access ---

    def beta_matrix_at(self, date: Union[pd.Timestamp, str]) -> pd.DataFrame:
        """(n_assets x n_factors) beta matrix at a specific date.

        Uses time-varying Kalman betas if available (``beta_ts`` in results),
        otherwise falls back to static ``beta_loadings``.
        """
        dt = pd.Timestamp(date)
        beta_ts = self.results.get("beta_ts")
        if beta_ts is None:
            return self.results["beta_loadings"].copy()

        asset_names = list(self.results["beta_loadings"].index)
        factor_names = list(self.results["beta_loadings"].columns)
        out = pd.DataFrame(0.0, index=asset_names, columns=factor_names)
        for a in asset_names:
            if a in beta_ts and dt in beta_ts[a].index:
                for f in beta_ts[a].columns:
                    if f in out.columns:
                        out.loc[a, f] = float(beta_ts[a].loc[dt, f])
        return out

    # --- Covariance reconstruction (pure linear algebra, no refit) ---

    def factor_cov_at(
        self,
        date: Union[pd.Timestamp, str],
        *,
        kind: str = "cond",
    ) -> pd.DataFrame:
        """Reconstruct factor covariance at *date* from eigenvectors + PC variances.

        Parameters
        ----------
        kind
            ``"cond"`` uses ``pc_cond_var`` (forecast from t-1 for day t).
            ``"forecast_next"`` uses ``pc_forecast_var`` (forecast at t for t+1).
        """
        dt = pd.Timestamp(date)
        ev = self.results["eigen_vectors"].to_numpy(dtype=float)

        pc_var_df = (self.results["pc_cond_var"] if kind == "cond"
                     else self.results["pc_forecast_var"])
        if dt not in pc_var_df.index:
            raise KeyError(f"date {dt} not found in pc variance index")

        h = np.clip(pc_var_df.loc[dt].to_numpy(dtype=float).reshape(-1), 0.0, None)
        cov = ev @ np.diag(h) @ ev.T
        cov = 0.5 * (cov + cov.T)

        names = list(self.results["eigen_vectors"].index)
        return pd.DataFrame(cov, index=names, columns=names)

    def factor_corr_at(
        self,
        date: Union[pd.Timestamp, str],
        *,
        kind: str = "cond",
    ) -> pd.DataFrame:
        """Factor correlation matrix at *date*."""
        return covariance_to_correlation_df(self.factor_cov_at(date, kind=kind), clip=True)

    def asset_cov_at(
        self,
        date: Union[pd.Timestamp, str],
        *,
        kind: str = "cond",
    ) -> pd.DataFrame:
        """Reconstruct asset covariance at *date*.

        Uses ``Sigma_a = (B @ V) diag(h_pc) (B @ V)' + diag(h_resid)``.
        When time-varying betas are available, uses ``beta_matrix_at(date)``.
        """
        dt = pd.Timestamp(date)

        if "beta_ts" in self.results:
            beta = self.beta_matrix_at(dt).to_numpy(dtype=float)
        else:
            beta = self.results["beta_loadings"].to_numpy(dtype=float)
        ev = self.results["eigen_vectors"].to_numpy(dtype=float)
        pc_exposure = beta @ ev

        pc_var_df = (self.results["pc_cond_var"] if kind == "cond"
                     else self.results["pc_forecast_var"])
        resid_var_df = (self.results["resid_cond_var"] if kind == "cond"
                        else self.results["resid_forecast_var"])

        if dt not in pc_var_df.index:
            raise KeyError(f"date {dt} not found in pc variance index")
        if dt not in resid_var_df.index:
            raise KeyError(f"date {dt} not found in resid variance index")

        h_pc = np.clip(pc_var_df.loc[dt].to_numpy(dtype=float).reshape(-1), 0.0, None)
        h_resid = np.clip(resid_var_df.loc[dt].to_numpy(dtype=float).reshape(-1), 0.0, None)

        scaled = pc_exposure * np.sqrt(h_pc).reshape(1, -1)
        cov = scaled @ scaled.T + np.diag(h_resid)
        cov = 0.5 * (cov + cov.T)

        names = list(self.results["beta_loadings"].index)
        return pd.DataFrame(cov, index=names, columns=names)

    def asset_corr_at(
        self,
        date: Union[pd.Timestamp, str],
        *,
        kind: str = "cond",
    ) -> pd.DataFrame:
        """Asset correlation matrix at *date*."""
        return covariance_to_correlation_df(self.asset_cov_at(date, kind=kind), clip=True)

    def iter_asset_covariances(
        self,
        *,
        dates: Optional[Sequence[pd.Timestamp]] = None,
        kind: str = "cond",
    ) -> Iterator[Tuple[pd.Timestamp, np.ndarray]]:
        """Yield ``(date, cov_array)`` without building a 3-D tensor.

        When time-varying betas are available, uses ``beta_matrix_at(date)``
        for each date.
        """
        pc_var_df = (self.results["pc_cond_var"] if kind == "cond"
                     else self.results["pc_forecast_var"])
        resid_var_df = (self.results["resid_cond_var"] if kind == "cond"
                        else self.results["resid_forecast_var"])

        idx = pc_var_df.index.intersection(resid_var_df.index)
        if dates is None:
            dates_iter = idx
        else:
            dates_iter = pd.DatetimeIndex(dates).intersection(idx)

        has_beta_ts = "beta_ts" in self.results
        ev = self.results["eigen_vectors"].to_numpy(dtype=float)

        if not has_beta_ts:
            beta_static = self.results["beta_loadings"].to_numpy(dtype=float)
            pc_exposure_static = beta_static @ ev

        for dt in dates_iter:
            h_pc = np.clip(pc_var_df.loc[dt].to_numpy(dtype=float).reshape(-1), 0.0, None)
            h_resid = np.clip(resid_var_df.loc[dt].to_numpy(dtype=float).reshape(-1), 0.0, None)

            if has_beta_ts:
                beta = self.beta_matrix_at(dt).to_numpy(dtype=float)
                pc_exposure = beta @ ev
            else:
                pc_exposure = pc_exposure_static

            scaled = pc_exposure * np.sqrt(h_pc).reshape(1, -1)
            cov = scaled @ scaled.T + np.diag(h_resid)
            cov = 0.5 * (cov + cov.T)
            yield pd.Timestamp(dt), cov

    # --- Plot convenience methods ---

    def plot_beta_heatmap(self, *, ax=None, **kwargs):
        """Heatmap of beta loadings."""
        from toolkit.risk_charts import plot_beta_heatmap
        betas = self.results.get("betas_ordered", self.beta_loadings)
        return plot_beta_heatmap(betas, ax=ax, **kwargs)

    def plot_factor_risk_heatmap(self, *, date=None, kind="cond",
                                 metric="correlation", ax=None, **kwargs):
        """Factor risk heatmap (correlation or covariance) at a given date."""
        from toolkit.risk_charts import plot_factor_risk_heatmap
        if date is None:
            cov_df = self["factor_cov_forecast"]
        else:
            cov_df = self.factor_cov_at(date, kind=kind)
        if metric == "correlation":
            matrix = covariance_to_correlation_df(cov_df, clip=True)
        else:
            matrix = cov_df
        return plot_factor_risk_heatmap(matrix, ax=ax, metric=metric, **kwargs)

    def plot_volatility_backtest(self, *, asset=None, annualize=True, ax=None, **kwargs):
        """Predicted vs realized trailing volatility for one asset."""
        from toolkit.risk_charts import plot_factor_volatility_backtest
        ev = self.results.get("evaluation")
        if not isinstance(ev, dict):
            raise ValueError("No evaluation found. Call evaluate_train_test() first.")
        ts = ev.get("timeseries", {})
        pred = ts.get("pred_vol", pd.DataFrame())
        real = ts.get("real_vol", pd.DataFrame())
        train_end = ev.get("params", {}).get("train_end")
        if asset is None:
            asset = str(pred.columns[0]) if not pred.empty else None
        return plot_factor_volatility_backtest(
            pred[asset] if asset else pred,
            real[asset] if asset else real,
            train_end=train_end,
            annualize=annualize, ax=ax, **kwargs,
        )

    def plot_returns_with_confidence_bands(self, *, asset=None, ax=None, **kwargs):
        """Daily returns with conditional volatility bands."""
        from toolkit.risk_charts import plot_returns_with_confidence_bands
        assets = self.assets
        if asset is None:
            asset = assets[0]
        r = self["resid"][asset].astype(float)
        s = np.sqrt(self["resid_cond_var"][asset].astype(float).clip(lower=0.0))
        train_end = self.meta.get("train_end")
        return plot_returns_with_confidence_bands(r, s, train_end=train_end,
                                                  ax=ax, **kwargs)

    def plot_aggregate_volatility_backtest(self, *, annualize=True, ax=None, **kwargs):
        """Average predicted vs trailing volatility across all assets."""
        from toolkit.risk_charts import plot_factor_volatility_backtest
        ev = self.results.get("evaluation")
        if not isinstance(ev, dict):
            raise ValueError("No evaluation found. Call evaluate_train_test() first.")
        ts = ev.get("timeseries", {})
        pred = ts.get("pred_avg_vol", pd.Series(dtype=float))
        real = ts.get("real_avg_vol", pd.Series(dtype=float))
        train_end = ev.get("params", {}).get("train_end")
        return plot_factor_volatility_backtest(
            pred, real, train_end=train_end, annualize=annualize,
            ax=ax, title="Average predicted vs trailing volatility", **kwargs,
        )

    def plot_volatility_regression_scatter(self, *, annualize=True, ax=None, **kwargs):
        """Scatter: predicted vs trailing vol (1 point per asset)."""
        from toolkit.risk_charts import plot_volatility_regression_scatter
        ev = self.results.get("evaluation")
        if not isinstance(ev, dict):
            raise ValueError("No evaluation found. Call evaluate_train_test() first.")
        ts = ev.get("timeseries", {})
        pred_vol = ts.get("pred_vol", pd.DataFrame())
        real_vol = ts.get("real_vol", pd.DataFrame())
        train_end = ev.get("params", {}).get("train_end")
        return plot_volatility_regression_scatter(
            pred_vol, real_vol, train_end=train_end,
            annualize=annualize, ax=ax, **kwargs,
        )

    def plot_agg_correlation_backtest(self, *, asset, ax=None, **kwargs):
        """Backtest corr(equal-weighted aggregate, asset)."""
        from toolkit.risk_charts import plot_agg_correlation_backtest
        ev = self.results.get("evaluation")
        if not isinstance(ev, dict):
            raise ValueError("No evaluation found. Call evaluate_train_test() first.")
        ts = ev.get("timeseries", {})
        pred = ts.get("pred_corr_to_agg", pd.DataFrame())
        real = ts.get("real_corr_to_agg", pd.DataFrame())
        train_end = ev.get("params", {}).get("train_end")
        return plot_agg_correlation_backtest(pred[asset], real[asset],
                                            train_end=train_end, asset=asset,
                                            ax=ax, **kwargs)

    def plot_asset_residuals_and_vol(self, *, asset=None, annualize=True,
                                     ax=None, **kwargs):
        """Standardized residuals and conditional vol for one asset."""
        from toolkit.risk_charts import plot_asset_residuals_and_vol
        assets = self.assets
        if asset is None:
            asset = assets[0]
        z = self["resid_std"][asset]
        vol = np.sqrt(self["resid_cond_var"][asset].clip(lower=0.0))
        return plot_asset_residuals_and_vol(z, vol, asset=asset,
                                            annualize=annualize, ax=ax, **kwargs)

    def summary(self) -> str:
        """Human-readable summary string."""
        m = self.meta
        return (
            "RiskModelRun\n"
            f"  as_of:    {m.get('as_of_date')}\n"
            f"  train_end:{m.get('train_end')}\n"
            f"  n_obs:    {m.get('n_obs')}\n"
            f"  n_train:  {m.get('n_train')}\n"
            f"  n_assets: {m.get('n_assets')}\n"
            f"  n_factors:{m.get('n_factors')}\n"
            f"  n_pcs:    {m.get('n_pcs')}\n"
        )


# ---------------------------------------------------------------------------
# Parallel execution helpers
# ---------------------------------------------------------------------------

def _fit_and_filter_garch_one(
    name: str,
    y_train: pd.Series,
    y_full: pd.Series,
    full_index: pd.Index,
    garch_kwargs: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Fit + filter GARCH for a single series. Suitable for parallel dispatch."""
    gp = _fit_garch(y_train, **garch_kwargs)
    out = _filter_garch(y_full, gp, full_index=full_index)
    return name, out


def _estimate_betas_one(
    asset_name: str,
    y: pd.Series,
    X: pd.DataFrame,
    beta_kwargs: Dict[str, Any],
) -> Tuple[str, Any, List[str], Dict[str, Any]]:
    """Run a single asset beta regression. Suitable for parallel dispatch."""
    res, sel, meta = _estimate_asset_betas(y=y, X=X, **beta_kwargs)
    return asset_name, res, sel, meta


def _kalman_filter_one(args):
    """Run Kalman filter for a single asset. Suitable for parallel dispatch."""
    asset_name, y, X, train_end_idx, kalman_kwargs, ols_info = args
    from toolkit.analysis.kalman_filter import KalmanFilter
    kf = KalmanFilter(**kalman_kwargs)
    X_with_const = X.copy()
    X_with_const.insert(0, "alpha", 1.0)
    ols_kw = ols_info if ols_info else {}
    kres = kf.filter(y, X_with_const, train_end_idx=train_end_idx, **ols_kw)
    return (asset_name, kres)


def _rolling_ols_betas(y: np.ndarray, X: np.ndarray, window: int = 60) -> np.ndarray:
    """(T x k) rolling OLS betas. NaN for t < window-1."""
    T, k = X.shape
    betas = np.full((T, k), np.nan, dtype=np.float64)
    reg = 1e-8 * np.eye(k, dtype=np.float64)
    for t in range(window - 1, T):
        Xw = X[t - window + 1:t + 1]
        yw = y[t - window + 1:t + 1]
        try:
            betas[t] = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)
        except np.linalg.LinAlgError:
            pass
    return betas


def _parallel_map(fn, args_list: list, max_workers: Optional[int] = None):
    """Run *fn(*args)* for each args tuple in *args_list* using ThreadPoolExecutor.

    Uses threads (not processes) to avoid the heavy process-spawn overhead on
    Windows.  The numerical libraries (arch, scipy, sklearn, numpy) release the
    GIL during their C-level computation, so threads still achieve meaningful
    parallelism here.

    Falls back to sequential execution if there is only one item.
    """
    import os
    if max_workers is None:
        max_workers = min(len(args_list), os.cpu_count() or 4)
    if max_workers <= 1 or len(args_list) <= 1:
        return [fn(*a) for a in args_list]
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: list = [None] * len(args_list)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(fn, *a): i for i, a in enumerate(args_list)
            }
            for fut in as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()
        return results
    except Exception:
        return [fn(*a) for a in args_list]


# ---------------------------------------------------------------------------
# RiskModel
# ---------------------------------------------------------------------------

class RiskModel:
    """PCA-based risk model with GARCH conditional variance.

    Data
    ----
    Accepts a MultiIndex ``uni`` DataFrame (groups ``"assets"``, ``"factors"``,
    ``"benchmarks"``) **or** explicit ``assets``, ``factors``, ``rf`` arguments
    passed to :meth:`run` / :meth:`evaluate_train_test`.

    Pipeline
    --------
    1. Per-asset factor regressions (LassoCV + post-OLS refit) for beta loadings
    2. PCA (SVD) on factor returns to extract principal components
    3. GARCH(1,1) conditional variance on each PC and residual (EWMA fallback)
    4. Reconstruct time-varying covariance from stored eigenvectors + variances

    Important outputs
    -----------------
    - ``beta_loadings``: per-asset factor betas
    - ``eigen_vectors``: PCA loadings (factor -> PC)
    - ``resid_cond_var`` / ``pc_cond_var``: one-step-ahead conditional variances
    - ``asset_cov_forecast``: latest next-day asset covariance matrix
    - ``evaluation``: optional train/test backtest (volatility + correlation)
    """

    def __init__(
        self,
        *,
        rf_name: str = "Rf",
        # Regression / Lasso
        max_exhaustive_factors: int = 12,
        lasso_cv_splits: int = 5,
        lasso_alphas: int = 100,
        lasso_max_iter: int = 50_000,
        lasso_coef_tol: float = 1e-10,
        # Post-OLS pruning
        tstat_cutoff: float = 4.0,
        min_factors: int = 0,
        # GARCH
        garch_dist: str = "t",
        garch_scale: float = 100.0,
        garch_min_obs: int = 100,
        ewma_lambda: float = 0.94,
        # PCA
        pca_demean: bool = False,
        pca_n_components: Optional[int] = None,
        # Kalman
        kalman_method: str = "lbfgs",
        kalman_maxiter: int = 50,
        kalman_fallback_to_ols: bool = True,
        kalman_q_scale: float = 0.01,
    ):
        self.rf_name = str(rf_name)
        self.max_exhaustive_factors = int(max_exhaustive_factors)
        self.lasso_cv_splits = int(lasso_cv_splits)
        self.lasso_alphas = int(lasso_alphas)
        self.lasso_max_iter = int(lasso_max_iter)
        self.lasso_coef_tol = float(lasso_coef_tol)
        self.tstat_cutoff = float(tstat_cutoff)
        self.min_factors = int(min_factors)
        self.garch_dist = str(garch_dist)
        self.garch_scale = float(garch_scale)
        self.garch_min_obs = int(garch_min_obs)
        self.ewma_lambda = float(ewma_lambda)
        self.pca_demean = bool(pca_demean)
        self.pca_n_components = (None if pca_n_components is None
                                 else int(pca_n_components))
        self.kalman_method = str(kalman_method)
        self.kalman_maxiter = int(kalman_maxiter)
        self.kalman_fallback_to_ols = bool(kalman_fallback_to_ols)
        self.kalman_q_scale = float(kalman_q_scale)

    # --- Public API ---

    def run(
        self,
        *,
        assets: Optional[pd.DataFrame] = None,
        factors: Optional[pd.DataFrame] = None,
        rf: Optional[pd.Series] = None,
        uni: Optional[pd.DataFrame] = None,
        progress: bool = True,
    ) -> RiskModelRun:
        """Fit the model on the full sample and return a :class:`RiskModelRun`."""
        assets_df, factors_df, rf_s = self._coerce_inputs(
            assets=assets, factors=factors, rf=rf, uni=uni)
        idx = (assets_df.index.intersection(factors_df.index)
               .intersection(rf_s.index).sort_values())
        assets_df = assets_df.loc[idx].astype(float)
        factors_df = factors_df.loc[idx].astype(float)
        rf_s = rf_s.loc[idx].astype(float)

        return self._fit_train_and_filter_full(
            assets=assets_df, factors=factors_df, rf=rf_s,
            train_end=None, progress=progress,
        )

    def evaluate_train_test(
        self,
        *,
        assets: Optional[pd.DataFrame] = None,
        factors: Optional[pd.DataFrame] = None,
        rf: Optional[pd.Series] = None,
        uni: Optional[pd.DataFrame] = None,
        train_end: Optional[Union[pd.Timestamp, str]] = None,
        train_fraction: float = 0.7,
        realized_window: int = 60,
        benchmark_lag: int = 1,
        progress: bool = True,
    ) -> RiskModelRun:
        """Fit on a training window and evaluate forecasts vs realized.

        Parameters
        ----------
        realized_window
            Trailing window length for the benchmark covariance.
        benchmark_lag
            Days to lag the benchmark window (1 avoids same-day look-ahead).
        """
        assets_df, factors_df, rf_s = self._coerce_inputs(
            assets=assets, factors=factors, rf=rf, uni=uni)
        idx = (assets_df.index.intersection(factors_df.index)
               .intersection(rf_s.index).sort_values())
        assets_df = assets_df.loc[idx].astype(float)
        factors_df = factors_df.loc[idx].astype(float)
        rf_s = rf_s.loc[idx].astype(float)

        if len(idx) < max(100, realized_window + benchmark_lag + 5):
            raise ValueError("Not enough data for evaluation (need >= ~100 rows).")

        if train_end is None:
            cut = int(max(10, min(len(idx) - 10,
                                  int(round(train_fraction * len(idx))))))
            train_end_ts = pd.Timestamp(idx[cut])
        else:
            train_end_ts = pd.Timestamp(train_end)
            if train_end_ts not in idx:
                pos = idx.searchsorted(train_end_ts)
                pos = int(min(max(pos, 1), len(idx) - 1))
                train_end_ts = pd.Timestamp(idx[pos])

        run = self._fit_train_and_filter_full(
            assets=assets_df, factors=factors_df, rf=rf_s,
            train_end=train_end_ts, progress=progress,
        )

        eval_out = self._evaluate_against_realized(
            run=run,
            assets_excess=assets_df.sub(rf_s, axis=0),
            train_end=train_end_ts,
            realized_window=int(realized_window),
            benchmark_lag=int(benchmark_lag),
            progress=progress,
        )

        run.results["evaluation"] = eval_out
        run.results["meta"]["train_end"] = train_end_ts
        run.results["meta"]["realized_window"] = int(realized_window)
        run.results["meta"]["benchmark_lag"] = int(benchmark_lag)
        return run

    # --- Input coercion ---

    def _coerce_inputs(
        self,
        *,
        assets: Optional[pd.DataFrame],
        factors: Optional[pd.DataFrame],
        rf: Optional[pd.Series],
        uni: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        if uni is not None:
            if not isinstance(uni.columns, pd.MultiIndex):
                raise TypeError(
                    "uni must have MultiIndex columns with groups: "
                    "assets, factors, benchmarks")
            if "assets" not in uni.columns.get_level_values(0):
                raise KeyError("uni must contain group 'assets'")
            if "factors" not in uni.columns.get_level_values(0):
                raise KeyError("uni must contain group 'factors'")
            if "benchmarks" not in uni.columns.get_level_values(0):
                raise KeyError("uni must contain group 'benchmarks'")
            if self.rf_name not in uni["benchmarks"].columns:
                raise KeyError(
                    f"uni['benchmarks'] must contain '{self.rf_name}'")

            assets_df = uni["assets"].copy().sort_index()
            factors_df = uni["factors"].copy().sort_index()
            rf_s = (uni["benchmarks"][self.rf_name].copy()
                    .sort_index().astype(float))
            return assets_df, factors_df, rf_s

        if assets is None or factors is None or rf is None:
            raise ValueError("Provide either uni=... OR (assets, factors, rf).")

        return (assets.copy().sort_index(),
                factors.copy().sort_index(),
                rf.copy().sort_index())

    # --- Core pipeline ---

    def _fit_train_and_filter_full(
        self,
        *,
        assets: pd.DataFrame,
        factors: pd.DataFrame,
        rf: pd.Series,
        train_end: Optional[pd.Timestamp],
        progress: bool,
    ) -> RiskModelRun:
        idx = (assets.index.intersection(factors.index)
               .intersection(rf.index).sort_values())
        assets = assets.loc[idx]
        factors = factors.loc[idx]
        rf = rf.loc[idx]

        if train_end is None:
            idx_train = idx
        else:
            idx_train = idx[idx < train_end]
            if len(idx_train) < 50:
                raise ValueError(
                    "Training window too short (< 50 rows).")

        assets_excess = assets.sub(rf, axis=0)
        assets_excess_train = assets_excess.loc[idx_train]
        factors_train = factors.loc[idx_train]

        asset_names = list(assets.columns)
        factor_names = list(factors.columns)

        # --- Step 1: betas (parallel) ---
        beta_loadings = pd.DataFrame(0.0, index=asset_names, columns=factor_names)
        alpha_intercept = pd.Series(np.nan, index=asset_names, name="alpha_intercept")
        r2 = pd.Series(np.nan, index=asset_names, name="r2")
        lasso_alpha = pd.Series(np.nan, index=asset_names, name="lasso_alpha")
        selected_factors: dict[str, List[str]] = {}
        by_asset_model: dict[str, Any] = {}

        beta_kwargs: Dict[str, Any] = dict(
            max_exhaustive_factors=self.max_exhaustive_factors,
            lasso_cv_splits=self.lasso_cv_splits,
            lasso_alphas=self.lasso_alphas,
            lasso_max_iter=self.lasso_max_iter,
            lasso_coef_tol=self.lasso_coef_tol,
            tstat_cutoff=self.tstat_cutoff,
            min_factors=self.min_factors,
        )
        beta_args = [
            (a, assets_excess_train[a].rename(a), factors_train, beta_kwargs)
            for a in asset_names
        ]
        beta_results = _parallel_map(_estimate_betas_one, beta_args)
        ols_results_by_asset: dict[str, Any] = {}
        for a, res, sel, meta in beta_results:
            selected_factors[a] = list(sel)
            lasso_alpha.loc[a] = float(meta.get("lasso_alpha", np.nan))
            alpha_intercept.loc[a] = float(res.params.get("alpha", np.nan))
            r2.loc[a] = float(getattr(res, "rsquared", np.nan))
            ols_results_by_asset[a] = res

            for k, v in res.params.items():
                if k != "alpha" and k in beta_loadings.columns:
                    beta_loadings.loc[a, k] = float(v)

            by_asset_model[a] = {
                "selected_factors": list(sel),
                "nobs": int(meta.get("nobs", np.nan) or getattr(res, "nobs", 0)),
                "r2": float(getattr(res, "rsquared", np.nan)),
                "adj_r2": float(getattr(res, "rsquared_adj", np.nan)),
                "bic": float(getattr(res, "bic", np.nan)),
                "tvalues": (pd.Series(getattr(res, "tvalues", pd.Series()),
                                      index=getattr(res, "params", pd.Series()).index)
                            .drop("alpha", errors="ignore")),
                "pvalues": (pd.Series(getattr(res, "pvalues", pd.Series()),
                                      index=getattr(res, "params", pd.Series()).index)
                            .drop("alpha", errors="ignore")),
                "lasso_meta": dict(meta),
            }

        # --- Step 1b: Kalman filter per asset (parallel) ---
        kalman_kwargs: Dict[str, Any] = dict(
            method=self.kalman_method,
            maxiter=self.kalman_maxiter,
            fallback_to_ols=self.kalman_fallback_to_ols,
            q_scale=self.kalman_q_scale,
        )
        train_end_idx_val = len(idx_train)

        # Extract OLS results to warm-start the Kalman filter
        ols_info_by_asset: dict[str, dict] = {}
        for a in asset_names:
            res = ols_results_by_asset.get(a)
            if res is not None and hasattr(res, "params") and hasattr(res, "cov_params"):
                try:
                    # res.params includes "alpha" as the first element,
                    # which maps to the ones column we prepend in _kalman_filter_one
                    ols_beta = res.params.to_numpy(dtype=np.float64)
                    ols_P = res.cov_params().to_numpy(dtype=np.float64)
                    ols_P = 0.5 * (ols_P + ols_P.T)
                    ols_var_resid = float(res.mse_resid) if hasattr(res, "mse_resid") else None
                    if ols_var_resid is not None:
                        ols_info_by_asset[a] = dict(
                            ols_beta=ols_beta, ols_P=ols_P,
                            ols_var_resid=ols_var_resid)
                except Exception:
                    pass

        kalman_args = [
            ((a,
              assets_excess[a].rename(a),
              factors[selected_factors[a]] if selected_factors[a] else pd.DataFrame(index=idx),
              train_end_idx_val,
              kalman_kwargs,
              ols_info_by_asset.get(a, {})),)
            for a in asset_names
        ]
        kalman_results = _parallel_map(_kalman_filter_one, kalman_args)

        beta_ts: dict[str, pd.DataFrame] = {}
        alpha_ts: dict[str, pd.Series] = {}
        beta_se: dict[str, pd.DataFrame] = {}
        kalman_diagnostics: dict[str, Any] = {}
        resid = pd.DataFrame(index=idx, columns=asset_names, dtype=float)

        for a, kres in _iter_with_progress(
                kalman_results, desc="Kalman filter",
                unit="asset", disable=not progress):
            # Extract time-varying betas (excluding alpha column)
            if "alpha" in kres.filtered_betas.columns:
                beta_ts[a] = kres.filtered_betas.drop(columns=["alpha"])
                alpha_ts[a] = kres.filtered_betas["alpha"]
                beta_se[a] = kres.filtered_se.drop(columns=["alpha"])
            else:
                beta_ts[a] = kres.filtered_betas
                alpha_ts[a] = pd.Series(0.0, index=kres.filtered_betas.index,
                                        name="alpha")
                beta_se[a] = kres.filtered_se

            kalman_diagnostics[a] = {
                "R": kres.R,
                "Q": kres.Q,
                "log_likelihood": kres.log_likelihood,
            }

            # Populate beta_loadings from last filtered betas (for downstream compat)
            last_betas = beta_ts[a].iloc[-1]
            for f in last_betas.index:
                if f in beta_loadings.columns:
                    beta_loadings.loc[a, f] = float(last_betas[f])

            # Populate alpha_intercept from last filtered alpha
            alpha_intercept.loc[a] = float(alpha_ts[a].iloc[-1])

            # Compute residuals using filtered betas: y - alpha_t - sum(beta_t * x_t)
            y_a = assets_excess[a].reindex(idx).to_numpy(dtype=float)
            alpha_arr = alpha_ts[a].reindex(idx).to_numpy(dtype=float)
            sel_factors = selected_factors[a]
            if sel_factors:
                X_sel = factors[sel_factors].reindex(idx).to_numpy(dtype=float)
                beta_arr = beta_ts[a].reindex(idx).to_numpy(dtype=float)
                pred = alpha_arr + np.sum(beta_arr * X_sel, axis=1)
            else:
                pred = alpha_arr
            resid[a] = y_a - pred

            # R2 from Kalman residuals on training window
            resid_train = resid[a].iloc[:train_end_idx_val].to_numpy(dtype=float)
            y_train = assets_excess[a].iloc[:train_end_idx_val].to_numpy(dtype=float)
            ss_res = float(np.nansum(resid_train ** 2))
            ss_tot = float(np.nansum((y_train - np.nanmean(y_train)) ** 2))
            r2.loc[a] = float(1.0 - ss_res / max(ss_tot, 1e-20))

        resid = resid.astype(float)

        # --- Step 2: residual GARCH (parallel) ---
        resid_cond_var = pd.DataFrame(index=idx, columns=asset_names, dtype=float)
        resid_forecast_var = pd.DataFrame(index=idx, columns=asset_names, dtype=float)
        resid_std = pd.DataFrame(index=idx, columns=asset_names, dtype=float)
        resid_garch_meta: dict[str, Any] = {}

        resid_garch_kwargs: Dict[str, Any] = dict(
            mean="Zero", dist=self.garch_dist, scale=self.garch_scale,
            min_obs=self.garch_min_obs, ewma_lambda=self.ewma_lambda,
        )
        resid_args = [
            (a, resid.loc[idx_train, a], resid[a], idx, resid_garch_kwargs)
            for a in asset_names
        ]
        resid_results = _parallel_map(_fit_and_filter_garch_one, resid_args)
        for a, out in resid_results:
            resid_cond_var[a] = out["cond_var"]
            resid_forecast_var[a] = out["forecast_var"]
            resid_std[a] = out["std_resid"]
            resid_garch_meta[a] = out["meta"]

        # --- Step 3: PCA + PC GARCH (parallel) ---
        eigen_vectors, singular_values = _fit_pca_svd(
            factors_train, demean=self.pca_demean,
            n_components=self.pca_n_components,
        )
        pc_returns = _transform_pca(factors, eigen_vectors,
                                    demean=self.pca_demean)

        pc_names = list(pc_returns.columns)
        pc_cond_var = pd.DataFrame(index=idx, columns=pc_names, dtype=float)
        pc_forecast_var = pd.DataFrame(index=idx, columns=pc_names, dtype=float)
        pc_std = pd.DataFrame(index=idx, columns=pc_names, dtype=float)
        pc_garch_meta: dict[str, Any] = {}

        pc_garch_kwargs: Dict[str, Any] = dict(
            mean="Constant", dist=self.garch_dist, scale=self.garch_scale,
            min_obs=self.garch_min_obs, ewma_lambda=self.ewma_lambda,
        )
        pc_args = [
            (pc, pc_returns.loc[idx_train, pc], pc_returns[pc], idx, pc_garch_kwargs)
            for pc in pc_names
        ]
        pc_results = _parallel_map(_fit_and_filter_garch_one, pc_args)
        for pc, out in pc_results:
            pc_cond_var[pc] = out["cond_var"]
            pc_forecast_var[pc] = out["forecast_var"]
            pc_std[pc] = out["std_resid"]
            pc_garch_meta[pc] = out["meta"]

        # Whitened factors (PC innovations -> factor basis)
        factors_whitened = pd.DataFrame(
            pc_std.to_numpy(dtype=float)
            @ eigen_vectors.to_numpy(dtype=float).T,
            index=idx, columns=eigen_vectors.index,
        )

        # Model-implied asset conditional variances (diagonal only)
        # Build time-varying beta array: (T, n_assets, n_factors) zero-padded
        ev_mat = eigen_vectors.to_numpy(dtype=float)
        n_pcs = ev_mat.shape[1]
        T_full = len(idx)
        n_a = len(asset_names)
        n_f = len(factor_names)
        beta_3d = np.zeros((T_full, n_a, n_f), dtype=np.float64)
        for ai, a in enumerate(asset_names):
            if a in beta_ts and not beta_ts[a].empty:
                for fi, f in enumerate(factor_names):
                    if f in beta_ts[a].columns:
                        beta_3d[:, ai, fi] = (
                            beta_ts[a][f].reindex(idx).to_numpy(dtype=float))

        # Time-varying pc_exposure: (T, n_assets, n_pcs)
        pc_exposure_3d = np.einsum('taf,fp->tap', beta_3d, ev_mat)

        # Vectorized conditional variance: sum_p (exposure_tip^2 * h_pc_tp)
        h_pc_arr = pc_cond_var.to_numpy(dtype=float)  # (T, n_pcs)
        asset_var_factor = np.einsum(
            'tip,tp->ti', pc_exposure_3d ** 2, h_pc_arr)  # (T, n_assets)

        asset_cond_var = pd.DataFrame(asset_var_factor, index=idx,
                                      columns=asset_names)
        asset_cond_var = (asset_cond_var.add(resid_cond_var, fill_value=0.0)
                          .clip(lower=0.0))
        asset_cond_vol = np.sqrt(asset_cond_var)

        # Latest next-day forecasts (use latest filtered betas)
        last_dt = pd.Timestamp(idx[-1])
        beta_latest = beta_loadings.to_numpy(dtype=float)  # from last filtered
        pc_exposure_latest = beta_latest @ ev_mat

        factor_cov_forecast = _factor_cov_from_pc_var(
            eigen_vectors=ev_mat,
            pc_var=np.clip(pc_forecast_var.loc[last_dt]
                           .to_numpy(dtype=float), 0.0, None),
            factor_names=list(eigen_vectors.index),
        )
        asset_cov_forecast = _asset_cov_from_vars(
            A=pc_exposure_latest,
            pc_var=np.clip(pc_forecast_var.loc[last_dt]
                           .to_numpy(dtype=float), 0.0, None),
            resid_var=np.clip(resid_forecast_var.loc[last_dt]
                              .to_numpy(dtype=float), 0.0, None),
            asset_names=asset_names,
        )

        betas_ordered = _order_betas_for_heatmap(
            beta_loadings=beta_loadings, factors=factors,
            r2=r2, selected_factors=selected_factors,
        )

        # --- Rolling OLS betas per asset ---
        rolling_window = 60
        rolling_ols_ts: dict[str, pd.DataFrame] = {}
        for a in asset_names:
            sel = selected_factors[a]
            if sel:
                y_a = assets_excess[a].reindex(idx).to_numpy(dtype=float)
                X_sel = factors[sel].reindex(idx).to_numpy(dtype=float)
                roll_betas = _rolling_ols_betas(y_a, X_sel, window=rolling_window)
                rolling_ols_ts[a] = pd.DataFrame(
                    roll_betas, index=idx, columns=sel)

        results: dict[str, Any] = {
            "meta": {
                "as_of_date": last_dt,
                "train_end": train_end,
                "n_obs": int(len(idx)),
                "n_train": int(len(idx_train)),
                "n_assets": int(len(asset_names)),
                "n_factors": int(len(factor_names)),
                "n_pcs": int(len(pc_names)),
                "rf_name": self.rf_name,
                "pca_demean": self.pca_demean,
                "pca_n_components": self.pca_n_components,
                "kalman_method": self.kalman_method,
                "kalman_maxiter": self.kalman_maxiter,
                "kalman_fallback_to_ols": self.kalman_fallback_to_ols,
            },
            # Step 1: LassoCV selection + Kalman betas
            "beta_loadings": beta_loadings,
            "alpha_intercept": alpha_intercept,
            "lasso_alpha": lasso_alpha,
            "r2": r2,
            "selected_factors": selected_factors,
            "betas_ordered": betas_ordered,
            "by_asset_model": by_asset_model,
            "assets_excess": assets_excess,
            # Kalman time-varying outputs
            "beta_ts": beta_ts,
            "alpha_ts": alpha_ts,
            "beta_se": beta_se,
            "kalman_diagnostics": kalman_diagnostics,
            "rolling_ols_ts": rolling_ols_ts,
            # Step 2
            "resid": resid,
            "resid_cond_var": resid_cond_var,
            "resid_forecast_var": resid_forecast_var,
            "resid_std": resid_std,
            "resid_garch_meta": resid_garch_meta,
            # Step 3
            "eigen_vectors": eigen_vectors,
            "singular_values": singular_values,
            "pc_returns": pc_returns,
            "pc_cond_var": pc_cond_var,
            "pc_forecast_var": pc_forecast_var,
            "pc_std": pc_std,
            "pc_garch_meta": pc_garch_meta,
            "factors_whitened": factors_whitened,
            # Risk outputs
            "factor_cov_forecast": factor_cov_forecast,
            "asset_cov_forecast": asset_cov_forecast,
            "asset_cond_var": asset_cond_var,
            "asset_cond_vol": asset_cond_vol,
        }

        params: dict[str, Any] = {
            "rf_name": self.rf_name,
            "max_exhaustive_factors": self.max_exhaustive_factors,
            "lasso_cv_splits": self.lasso_cv_splits,
            "lasso_alphas": self.lasso_alphas,
            "lasso_max_iter": self.lasso_max_iter,
            "lasso_coef_tol": self.lasso_coef_tol,
            "tstat_cutoff": self.tstat_cutoff,
            "min_factors": self.min_factors,
            "garch_dist": self.garch_dist,
            "garch_scale": self.garch_scale,
            "garch_min_obs": self.garch_min_obs,
            "ewma_lambda": self.ewma_lambda,
            "pca_demean": self.pca_demean,
            "pca_n_components": self.pca_n_components,
            "kalman_method": self.kalman_method,
            "kalman_maxiter": self.kalman_maxiter,
            "kalman_fallback_to_ols": self.kalman_fallback_to_ols,
        }

        return RiskModelRun(params=params, results=results)

    # --- Evaluation engine ---

    def _evaluate_against_realized(
        self,
        *,
        run: RiskModelRun,
        assets_excess: pd.DataFrame,
        train_end: pd.Timestamp,
        realized_window: int,
        benchmark_lag: int,
        progress: bool,
    ) -> dict[str, Any]:
        """Compare predicted risk vs a trailing-window benchmark.

        Evaluates volatility (sqrt(var)) and correlation (scale-free).
        """
        idx = (assets_excess.index
               .intersection(run["asset_cond_var"].index).sort_values())
        assets_excess = assets_excess.loc[idx]
        w = int(realized_window)
        lag = int(benchmark_lag)
        if lag < 0:
            raise ValueError("benchmark_lag must be >= 0")
        if w <= 2:
            raise ValueError("realized_window must be > 2")

        start_pos = w + lag - 1
        eval_dates = idx[start_pos:]

        asset_names = list(run["beta_loadings"].index)
        assets_excess = assets_excess.reindex(columns=asset_names)

        X = assets_excess.to_numpy(dtype=float)
        n_assets = X.shape[1]
        n_eval = len(eval_dates)
        w_agg = np.full(n_assets, 1.0 / float(n_assets), dtype=float)

        beta = run["beta_loadings"].to_numpy(dtype=float)
        ev = run["eigen_vectors"].to_numpy(dtype=float)
        pc_exposure = beta @ ev

        pc_var_arr = run["pc_cond_var"].reindex(index=idx).to_numpy(dtype=float)
        resid_var_arr = run["resid_cond_var"].reindex(index=idx).to_numpy(dtype=float)

        # Pre-compute integer positions for eval_dates within idx
        eval_positions = np.array([int(idx.get_loc(dt)) for dt in eval_dates])

        # Pairwise upper-triangle indices
        triu_i, triu_j = np.triu_indices(n_assets, k=1)
        n_pairs = len(triu_i)
        pair_keys: List[str] = []
        for pi in range(n_pairs):
            a, b = sorted([asset_names[triu_i[pi]], asset_names[triu_j[pi]]])
            pair_keys.append(f"{a}|{b}")

        # Pre-allocate numpy arrays instead of DataFrames/Series
        vol_mse_arr = np.full(n_eval, np.nan, dtype=float)
        corr_frob_arr = np.full(n_eval, np.nan, dtype=float)
        pred_vol_arr = np.full((n_eval, n_assets), np.nan, dtype=float)
        real_vol_arr = np.full((n_eval, n_assets), np.nan, dtype=float)
        pred_corr_agg_arr = np.full((n_eval, n_assets), np.nan, dtype=float)
        real_corr_agg_arr = np.full((n_eval, n_assets), np.nan, dtype=float)
        pred_cov_agg_arr = np.full((n_eval, n_assets), np.nan, dtype=float)
        real_cov_agg_arr = np.full((n_eval, n_assets), np.nan, dtype=float)
        pred_corr_pw_arr = np.full((n_eval, n_pairs), np.nan, dtype=float)
        real_corr_pw_arr = np.full((n_eval, n_pairs), np.nan, dtype=float)
        pred_cov_pw_arr = np.full((n_eval, n_pairs), np.nan, dtype=float)
        real_cov_pw_arr = np.full((n_eval, n_pairs), np.nan, dtype=float)

        min_rows = max(5, w // 2)

        for ti, dt in enumerate(_iter_with_progress(
                eval_dates,
                desc="Eval: trailing vol/corr vs model",
                unit="day", disable=not progress)):
            j = eval_positions[ti]
            end = j - lag
            start = end - w + 1
            Xw = X[start:(end + 1), :]
            Xw = Xw[np.isfinite(Xw).all(axis=1)]
            if Xw.shape[0] < min_rows:
                continue

            realized_cov = np.cov(Xw, rowvar=False, ddof=1)
            realized_cov = 0.5 * (realized_cov + realized_cov.T)

            h_pc = np.clip(pc_var_arr[j], 0.0, None)
            h_resid = np.clip(resid_var_arr[j], 0.0, None)

            scaled = pc_exposure * np.sqrt(h_pc).reshape(1, -1)
            pred_cov = scaled @ scaled.T + np.diag(h_resid)
            pred_cov = 0.5 * (pred_cov + pred_cov.T)

            pred_sigma = np.sqrt(np.clip(np.diag(pred_cov), 0.0, None))
            real_sigma = np.sqrt(np.clip(np.diag(realized_cov), 0.0, None))

            pred_vol_arr[ti] = pred_sigma
            real_vol_arr[ti] = real_sigma

            mask = np.isfinite(pred_sigma) & np.isfinite(real_sigma)
            if mask.any():
                diff = pred_sigma[mask] - real_sigma[mask]
                vol_mse_arr[ti] = float(np.mean(diff * diff))

            # Correlation backtest: corr(equal-weighted agg, asset)
            pred_cov_agg_vec = pred_cov @ w_agg
            real_cov_agg_vec = realized_cov @ w_agg
            pred_var_agg = float(w_agg @ pred_cov_agg_vec)
            real_var_agg = float(w_agg @ real_cov_agg_vec)
            pred_vol_agg = float(np.sqrt(max(pred_var_agg, 0.0)))
            real_vol_agg = float(np.sqrt(max(real_var_agg, 0.0)))

            pred_corr_vec = np.full(n_assets, np.nan, dtype=float)
            real_corr_vec = np.full(n_assets, np.nan, dtype=float)

            if pred_vol_agg > 1e-12:
                denom = pred_vol_agg * pred_sigma
                ok = np.isfinite(pred_cov_agg_vec) & np.isfinite(denom) & (denom > 1e-12)
                pred_corr_vec[ok] = pred_cov_agg_vec[ok] / denom[ok]

            if real_vol_agg > 1e-12:
                denom = real_vol_agg * real_sigma
                ok = np.isfinite(real_cov_agg_vec) & np.isfinite(denom) & (denom > 1e-12)
                real_corr_vec[ok] = real_cov_agg_vec[ok] / denom[ok]

            pred_corr_agg_arr[ti] = pred_corr_vec
            real_corr_agg_arr[ti] = real_corr_vec

            # Covariance backtest: cov(asset, equal-weighted agg)
            pred_cov_agg_arr[ti] = pred_cov_agg_vec
            real_cov_agg_arr[ti] = real_cov_agg_vec

            pred_corr = covariance_to_correlation(pred_cov, clip=True)
            real_corr = covariance_to_correlation(realized_cov, clip=True)
            corr_frob_arr[ti] = float(
                np.linalg.norm(pred_corr - real_corr, ord="fro"))

            # Pairwise correlations and covariances — vectorized via triu indices
            pred_corr_pw_arr[ti] = pred_corr[triu_i, triu_j]
            real_corr_pw_arr[ti] = real_corr[triu_i, triu_j]
            pred_cov_pw_arr[ti] = pred_cov[triu_i, triu_j]
            real_cov_pw_arr[ti] = realized_cov[triu_i, triu_j]

        # Convert numpy arrays to pandas objects
        vol_mse = pd.Series(vol_mse_arr, index=eval_dates, name="vol_mse")
        corr_frob_err = pd.Series(corr_frob_arr, index=eval_dates, name="corr_frob_err")
        pred_vol = pd.DataFrame(pred_vol_arr, index=eval_dates, columns=asset_names)
        real_vol = pd.DataFrame(real_vol_arr, index=eval_dates, columns=asset_names)
        pred_corr_to_agg = pd.DataFrame(pred_corr_agg_arr, index=eval_dates, columns=asset_names)
        real_corr_to_agg = pd.DataFrame(real_corr_agg_arr, index=eval_dates, columns=asset_names)
        pred_cov_to_agg = pd.DataFrame(pred_cov_agg_arr, index=eval_dates, columns=asset_names)
        real_cov_to_agg = pd.DataFrame(real_cov_agg_arr, index=eval_dates, columns=asset_names)

        pred_corr_pairwise: dict[str, pd.Series] = {}
        real_corr_pairwise: dict[str, pd.Series] = {}
        pred_cov_pairwise: dict[str, pd.Series] = {}
        real_cov_pairwise: dict[str, pd.Series] = {}
        for p, key in enumerate(pair_keys):
            pred_corr_pairwise[key] = pd.Series(pred_corr_pw_arr[:, p], index=eval_dates, dtype=float)
            real_corr_pairwise[key] = pd.Series(real_corr_pw_arr[:, p], index=eval_dates, dtype=float)
            pred_cov_pairwise[key] = pd.Series(pred_cov_pw_arr[:, p], index=eval_dates, dtype=float)
            real_cov_pairwise[key] = pd.Series(real_cov_pw_arr[:, p], index=eval_dates, dtype=float)

        ins = vol_mse.index[vol_mse.index < train_end]
        oos = vol_mse.index[vol_mse.index >= train_end]

        def _summ(s: pd.Series) -> dict[str, float]:
            s2 = s.dropna()
            return {
                "mean": float(s2.mean()) if len(s2) else np.nan,
                "median": float(s2.median()) if len(s2) else np.nan,
                "n": float(len(s2)),
            }

        summary = {
            "vol_mse_in_sample": _summ(vol_mse.loc[ins]),
            "vol_mse_out_of_sample": _summ(vol_mse.loc[oos]),
            "corr_frob_err_in_sample": _summ(corr_frob_err.loc[ins]),
            "corr_frob_err_out_of_sample": _summ(corr_frob_err.loc[oos]),
        }

        pred_avg_vol = pred_vol.mean(axis=1).rename("pred_avg_vol")
        real_avg_vol = real_vol.mean(axis=1).rename("real_avg_vol")

        pred_vol_ann = annualize_vol(pred_vol)
        real_vol_ann = annualize_vol(real_vol)
        pred_avg_vol_ann = annualize_vol(pred_avg_vol).rename("pred_avg_vol_ann")
        real_avg_vol_ann = annualize_vol(real_avg_vol).rename("real_avg_vol_ann")

        return {
            "params": {
                "train_end": pd.Timestamp(train_end),
                "realized_window": int(realized_window),
                "benchmark_lag": int(lag),
                "n_assets": int(n_assets),
            },
            "summary": summary,
            "timeseries": {
                "vol_mse": vol_mse,
                "corr_frob_err": corr_frob_err,
                "pred_vol": pred_vol,
                "real_vol": real_vol,
                "pred_avg_vol": pred_avg_vol,
                "real_avg_vol": real_avg_vol,
                "pred_vol_ann": pred_vol_ann,
                "real_vol_ann": real_vol_ann,
                "pred_avg_vol_ann": pred_avg_vol_ann,
                "real_avg_vol_ann": real_avg_vol_ann,
                "pred_corr_to_agg": pred_corr_to_agg,
                "real_corr_to_agg": real_corr_to_agg,
                "pred_corr_pairwise": pred_corr_pairwise,
                "real_corr_pairwise": real_corr_pairwise,
                "pred_cov_to_agg": pred_cov_to_agg,
                "real_cov_to_agg": real_cov_to_agg,
                "pred_cov_pairwise": pred_cov_pairwise,
                "real_cov_pairwise": real_cov_pairwise,
            },
        }


# ---------------------------------------------------------------------------
# Internal reconstruction helpers (single-matrix)
# ---------------------------------------------------------------------------

def _factor_cov_from_pc_var(
    *,
    eigen_vectors: np.ndarray,
    pc_var: np.ndarray,
    factor_names: List[str],
) -> pd.DataFrame:
    pc_var = np.clip(np.asarray(pc_var, dtype=float).reshape(-1), 0.0, None)
    Sf = eigen_vectors @ np.diag(pc_var) @ eigen_vectors.T
    Sf = 0.5 * (Sf + Sf.T)
    return pd.DataFrame(Sf, index=factor_names, columns=factor_names)


def _asset_cov_from_vars(
    *,
    A: np.ndarray,
    pc_var: np.ndarray,
    resid_var: np.ndarray,
    asset_names: List[str],
) -> pd.DataFrame:
    pc_var = np.clip(np.asarray(pc_var, dtype=float).reshape(-1), 0.0, None)
    resid_var = np.clip(np.asarray(resid_var, dtype=float).reshape(-1), 0.0, None)
    S = A * np.sqrt(pc_var).reshape(1, -1)
    cov = S @ S.T + np.diag(resid_var)
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=asset_names, columns=asset_names)


def _order_betas_for_heatmap(
    *,
    beta_loadings: pd.DataFrame,
    factors: pd.DataFrame,
    r2: pd.Series,
    selected_factors: dict[str, List[str]],
) -> pd.DataFrame:
    """Reorder beta matrix for heatmap readability."""
    used = sorted({f for lst in selected_factors.values() for f in lst})
    betas_used = beta_loadings[used].copy() if used else beta_loadings.copy()

    factor_cov = factors.cov()
    fvar = pd.Series(np.diag(factor_cov.to_numpy(dtype=float)),
                     index=factor_cov.index)
    fvar = fvar.reindex(betas_used.columns).fillna(0.0)
    score = (betas_used.pow(2).sum(axis=0) * fvar).sort_values(ascending=False)
    if len(score) > 0:
        betas_used = betas_used[score.index]

    if r2.notna().any():
        betas_used = betas_used.reindex(r2.sort_values(ascending=False).index)

    return betas_used


__all__ = [
    "RiskModel",
    "RiskModelRun",
    "FactorModel",
    "FactorRun",
    "annualize_vol",
    "covariance_to_correlation",
    "covariance_to_correlation_df",
]

# Backward compatibility aliases
FactorModel = RiskModel
FactorRun = RiskModelRun
