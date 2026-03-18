"""Kalman filter for time-varying parameter regression via statsmodels.

Uses ``statsmodels.tsa.statespace.MLEModel`` to implement the state-space
model for each asset:

    Observation:  r_{i,t} = f_t' beta_{i,t} + eps_{i,t},   eps ~ N(0, R)
    State:        beta_{i,t} = beta_{i,t-1} + eta_{i,t},    eta ~ N(0, Q)

statsmodels handles:
    - Exact MLE for hyperparameters (Q diagonal, R scalar)
    - Numerically stable square-root Kalman filter
    - Rauch-Tung-Striebel smoother
    - Log-likelihood evaluation

We expose a simple ``KalmanFilter`` class that wraps this into the
per-asset interface needed by the factor model pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel


@dataclass(frozen=True)
class KalmanResult:
    """Output container for a single-asset Kalman filter run.

    Attributes
    ----------
    filtered_betas : pd.DataFrame
        (T x k) filtered state estimates beta_{t|t}.
    smoothed_betas : pd.DataFrame
        (T x k) RTS-smoothed state estimates beta_{t|T}.
    predicted_betas : pd.DataFrame
        (T x k) one-step-ahead predictions beta_{t|t-1}.
    filtered_se : pd.DataFrame
        (T x k) filtered standard errors sqrt(diag(P_{t|t})).
    smoothed_se : pd.DataFrame
        (T x k) smoothed standard errors sqrt(diag(P_{t|T})).
    innovations : pd.Series
        (T,) one-step-ahead prediction errors.
    innovation_var : pd.Series
        (T,) innovation variances.
    log_likelihood : float
        Total log-likelihood from the Kalman filter.
    R : float
        Estimated observation noise variance.
    Q : np.ndarray
        (k x k) estimated state noise covariance (diagonal).
    """
    filtered_betas: pd.DataFrame
    smoothed_betas: pd.DataFrame
    predicted_betas: pd.DataFrame
    filtered_se: pd.DataFrame
    smoothed_se: pd.DataFrame
    innovations: pd.Series
    innovation_var: pd.Series
    log_likelihood: float
    R: float
    Q: np.ndarray


class TimeVaryingBetaModel(MLEModel):
    """State-space model for time-varying regression coefficients.

    State vector: beta_t (k x 1)
    Observation:  r_t = Z_t beta_t + eps_t,  Z_t = f_t' (1 x k)
    Transition:   beta_t = I beta_{t-1} + eta_t  (random walk)

    Free parameters (estimated via MLE):
        - log(q_1), ..., log(q_k): diagonal of Q (state noise)
        - log(R): observation noise variance
    """

    def __init__(self, endog: np.ndarray, exog: np.ndarray, **kwargs):
        T, k = exog.shape
        super().__init__(endog, k_states=k, k_posdef=k, **kwargs)

        self._exog = exog.copy()
        self._start_params_override = None

        # Transition = Identity (random walk)
        self["transition"] = np.eye(k, dtype=np.float64)
        # Selection = Identity (state noise enters all states)
        self["selection"] = np.eye(k, dtype=np.float64)

        # Time-varying design matrix: Z_t = f_t' (1 x k) at each t.
        design = np.zeros((1, k, T), dtype=np.float64)
        for t in range(T):
            design[0, :, t] = exog[t]
        self["design"] = design

    def initialize_from_ols(self, beta_ols: np.ndarray, P_ols: np.ndarray):
        """Set known initial state from OLS warm-start."""
        self.ssm.initialize_known(beta_ols, P_ols)

    @property
    def param_names(self):
        k = self.k_states
        return [f"log_q{i}" for i in range(k)] + ["log_R"]

    @property
    def start_params(self):
        if self._start_params_override is not None:
            return self._start_params_override
        k = self.k_states
        var_y = float(np.nanvar(self.endog)) if len(self.endog) > 0 else 1.0
        q_start = np.full(k, np.log(max(var_y * 0.1, 1e-10)))
        r_start = np.array([np.log(max(var_y * 0.5, 1e-10))])
        return np.concatenate([q_start, r_start])

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        k = self.k_states
        q_diag = np.exp(params[:k])
        self["state_cov"] = np.diag(q_diag)
        R = np.exp(params[k])
        self["obs_cov", 0, 0] = R


class KalmanFilter:
    """Kalman filter for time-varying regression coefficients.

    Wraps ``statsmodels`` state-space infrastructure for robust MLE
    estimation of Q and R, with built-in filtering and smoothing.

    Parameters
    ----------
    method : str
        Optimisation method for MLE ('lbfgs', 'powell', 'nm', etc.).
    maxiter : int
        Maximum iterations for the optimiser.
    fallback_to_ols : bool
        If MLE fails, fall back to a simple OLS estimate with constant
        betas (equivalent to Q = 0).
    """

    def __init__(
        self,
        *,
        method: str = "lbfgs",
        maxiter: int = 200,
        fallback_to_ols: bool = True,
        q_scale: float = 1.0,
    ):
        self.method = str(method)
        self.maxiter = int(maxiter)
        self.fallback_to_ols = bool(fallback_to_ols)
        self.q_scale = float(q_scale)

    def filter(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        *,
        train_end_idx: Optional[int] = None,
        ols_beta: Optional[np.ndarray] = None,
        ols_P: Optional[np.ndarray] = None,
        ols_var_resid: Optional[float] = None,
    ) -> KalmanResult:
        """Run the Kalman filter on one asset.

        Parameters
        ----------
        y : pd.Series
            (T,) asset excess returns.
        X : pd.DataFrame
            (T x k) factor returns (with intercept column if desired),
            aligned with *y*.
        train_end_idx : int, optional
            If provided, fit MLE on ``y[:train_end_idx]`` and
            ``X[:train_end_idx]``, then run the filter/smoother on the
            full sample with those fixed parameters.

        Returns
        -------
        KalmanResult
        """
        idx = y.index.intersection(X.index).sort_values()
        y = y.reindex(idx).astype(float)
        X = X.reindex(idx).astype(float)

        yv = y.to_numpy(dtype=np.float64).copy()
        Xv = X.to_numpy(dtype=np.float64).copy()
        T, k = Xv.shape
        factor_names = list(X.columns)

        _ols_kw = dict(ols_beta=ols_beta, ols_P=ols_P,
                       ols_var_resid=ols_var_resid)
        try:
            if train_end_idx is not None and 0 < train_end_idx < T:
                result = self._fit_train_apply_full(
                    yv, Xv, k, train_end_idx, **_ols_kw)
            else:
                result = self._fit_statespace(yv, Xv, k, **_ols_kw)
        except Exception:
            if self.fallback_to_ols:
                return self._ols_fallback(yv, Xv, idx, factor_names)
            raise

        # Extract states
        filtered = result.filtered_state.T        # (T, k)
        smoothed = result.smoothed_state.T         # (T, k)
        predicted = result.predicted_state[:, :T].T  # (T, k)

        # Filtered and smoothed standard errors from state covariances
        filt_cov = result.filtered_state_cov   # (k, k, T)
        filt_diag = np.diagonal(filt_cov, axis1=0, axis2=1)  # (T, k)
        filt_se = np.sqrt(np.clip(filt_diag, 0.0, None))     # (T, k)

        smooth_cov = result.smoothed_state_cov  # (k, k, T)
        smooth_diag = np.diagonal(smooth_cov, axis1=0, axis2=1)  # (T, k)
        smooth_se = np.sqrt(np.clip(smooth_diag, 0.0, None))     # (T, k)

        # Innovations and their variances
        innovations = result.forecasts_error[0, :]       # (T,)
        innov_var = result.forecasts_error_cov[0, 0, :]  # (T,)

        # Extract estimated Q and R
        params = result.params
        Q = np.diag(np.exp(params[:k]))
        R = float(np.exp(params[k]))

        return KalmanResult(
            filtered_betas=pd.DataFrame(filtered, index=idx, columns=factor_names),
            smoothed_betas=pd.DataFrame(smoothed, index=idx, columns=factor_names),
            predicted_betas=pd.DataFrame(predicted, index=idx, columns=factor_names),
            filtered_se=pd.DataFrame(filt_se, index=idx, columns=factor_names),
            smoothed_se=pd.DataFrame(smooth_se, index=idx, columns=factor_names),
            innovations=pd.Series(innovations, index=idx, name="innovation"),
            innovation_var=pd.Series(innov_var, index=idx, name="innovation_var"),
            log_likelihood=float(result.llf),
            R=R,
            Q=Q,
        )

    def _fit_statespace(self, yv, Xv, k, *, ols_beta=None, ols_P=None,
                        ols_var_resid=None):
        """Build and fit the state-space model with OLS-informed initialization."""
        if ols_beta is not None and ols_P is not None and ols_var_resid is not None:
            beta_ols, var_resid, P_ols = ols_beta, ols_var_resid, ols_P
        else:
            beta_ols, var_resid, P_ols = self._ols_warmup(yv, Xv, k)

        mod = TimeVaryingBetaModel(yv, Xv)
        mod.initialize_from_ols(beta_ols, P_ols)

        Q_start = np.diag(P_ols) * 0.01 * self.q_scale
        Q_start = np.clip(Q_start, 1e-12, None)
        R_start = max(var_resid, 1e-12)
        start_params = np.concatenate([
            np.log(Q_start),
            [np.log(R_start)],
        ])
        mod._start_params_override = start_params

        result = mod.fit(
            method=self.method,
            maxiter=self.maxiter,
            disp=False,
            start_params=start_params,
        )
        return result

    def _fit_train_apply_full(self, yv, Xv, k, train_end_idx, *,
                              ols_beta=None, ols_P=None, ols_var_resid=None):
        """Fit MLE on training data, then filter/smooth over full sample."""
        yv_train = yv[:train_end_idx]
        Xv_train = Xv[:train_end_idx]

        # Fit on training data
        if ols_beta is not None and ols_P is not None and ols_var_resid is not None:
            beta_ols, var_resid, P_ols = ols_beta, ols_var_resid, ols_P
        else:
            beta_ols, var_resid, P_ols = self._ols_warmup(yv_train, Xv_train, k)

        mod_train = TimeVaryingBetaModel(yv_train, Xv_train)
        mod_train.initialize_from_ols(beta_ols, P_ols)

        Q_start = np.diag(P_ols) * 0.01
        Q_start = np.clip(Q_start, 1e-12, None)
        R_start = max(var_resid, 1e-12)
        start_params = np.concatenate([
            np.log(Q_start),
            [np.log(R_start)],
        ])
        mod_train._start_params_override = start_params

        res_train = mod_train.fit(
            method=self.method,
            maxiter=self.maxiter,
            disp=False,
            start_params=start_params,
        )
        fitted_params = res_train.params.copy()

        # Scale Q (process noise) — lower q_scale = smoother betas
        if self.q_scale != 1.0:
            fitted_params[:k] += np.log(self.q_scale)

        # Apply fixed params on full data
        mod_full = TimeVaryingBetaModel(yv, Xv)
        mod_full.initialize_from_ols(beta_ols, P_ols)

        result = mod_full.smooth(fitted_params)
        return result

    @staticmethod
    def _ols_warmup(yv, Xv, k):
        """Compute OLS estimates for Kalman filter initialization."""
        T = len(yv)
        reg = 1e-8 * np.eye(k, dtype=np.float64)
        XtX = Xv.T @ Xv

        try:
            beta_ols = np.linalg.solve(XtX + reg, Xv.T @ yv)
        except np.linalg.LinAlgError:
            beta_ols = np.zeros(k, dtype=np.float64)

        resid = yv - Xv @ beta_ols
        var_resid = float(np.var(resid, ddof=min(k, T - 1)))
        var_resid = max(var_resid, 1e-10)

        try:
            P_ols = var_resid * np.linalg.inv(XtX + reg)
        except np.linalg.LinAlgError:
            P_ols = var_resid * np.eye(k, dtype=np.float64)
        P_ols = 0.5 * (P_ols + P_ols.T)

        return beta_ols, var_resid, P_ols

    @staticmethod
    def _ols_fallback(
        yv: np.ndarray,
        Xv: np.ndarray,
        idx: pd.DatetimeIndex,
        factor_names: list,
    ) -> KalmanResult:
        """OLS fallback: static betas broadcast across all dates."""
        T, k = Xv.shape
        reg = 1e-6 * np.eye(k)
        try:
            beta = np.linalg.solve(Xv.T @ Xv + reg, Xv.T @ yv)
        except np.linalg.LinAlgError:
            beta = np.zeros(k)

        resid = yv - Xv @ beta
        R = float(np.var(resid, ddof=k)) if T > k else 1.0

        betas_static = np.tile(beta, (T, 1))
        zeros_se = np.zeros((T, k))

        return KalmanResult(
            filtered_betas=pd.DataFrame(betas_static, index=idx, columns=factor_names),
            smoothed_betas=pd.DataFrame(betas_static, index=idx, columns=factor_names),
            predicted_betas=pd.DataFrame(betas_static, index=idx, columns=factor_names),
            filtered_se=pd.DataFrame(zeros_se, index=idx, columns=factor_names),
            smoothed_se=pd.DataFrame(zeros_se, index=idx, columns=factor_names),
            innovations=pd.Series(resid, index=idx, name="innovation"),
            innovation_var=pd.Series(np.full(T, R), index=idx, name="innovation_var"),
            log_likelihood=float(-0.5 * T * (np.log(2 * np.pi) + np.log(max(R, 1e-12)) + 1)),
            R=R,
            Q=np.zeros((k, k)),
        )


__all__ = [
    "KalmanFilter",
    "KalmanResult",
    "TimeVaryingBetaModel",
]
