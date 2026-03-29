"""Tests for the Kalman filter module."""

import numpy as np
import pandas as pd
import pytest

from toolkit.analysis.kalman_filter import KalmanFilter, KalmanResult


def _make_synthetic_data(T=500, k=3, seed=42):
    """Generate synthetic time-varying beta data."""
    rng = np.random.default_rng(seed)

    beta_true = np.zeros((T, k))
    beta_true[0] = rng.normal(0.5, 0.1, k)
    for t in range(1, T):
        beta_true[t] = beta_true[t - 1] + rng.normal(0, 0.005, k)

    X = rng.normal(0, 0.01, (T, k))
    noise = rng.normal(0, 0.005, T)
    y = np.sum(X * beta_true, axis=1) + noise

    dates = pd.bdate_range("2020-01-01", periods=T)
    factor_names = [f"F{i}" for i in range(k)]

    y_series = pd.Series(y, index=dates, name="asset")
    X_df = pd.DataFrame(X, index=dates, columns=factor_names)

    return y_series, X_df, beta_true


class TestKalmanFilter:

    def test_basic_output_shape(self):
        y, X, _ = _make_synthetic_data(T=200, k=3)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)

        assert isinstance(result, KalmanResult)
        assert result.filtered_betas.shape == (200, 3)
        assert result.smoothed_betas.shape == (200, 3)
        assert result.predicted_betas.shape == (200, 3)
        assert result.filtered_se.shape == (200, 3)
        assert result.smoothed_se.shape == (200, 3)
        assert len(result.innovations) == 200
        assert len(result.innovation_var) == 200

    def test_index_alignment(self):
        y, X, _ = _make_synthetic_data(T=200, k=2)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)
        pd.testing.assert_index_equal(result.filtered_betas.index, y.index)
        pd.testing.assert_index_equal(result.filtered_se.index, y.index)

    def test_column_names(self):
        y, X, _ = _make_synthetic_data(T=200, k=3)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)
        assert list(result.filtered_betas.columns) == ["F0", "F1", "F2"]
        assert list(result.filtered_se.columns) == ["F0", "F1", "F2"]

    def test_innovation_var_positive(self):
        y, X, _ = _make_synthetic_data(T=200, k=2)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)
        assert (result.innovation_var > 0).all()

    def test_R_positive(self):
        y, X, _ = _make_synthetic_data(T=200, k=2)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)
        assert result.R > 0

    def test_Q_positive_semidefinite(self):
        y, X, _ = _make_synthetic_data(T=200, k=3)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)
        eigvals = np.linalg.eigvalsh(result.Q)
        assert np.all(eigvals >= -1e-10)

    def test_log_likelihood_finite(self):
        y, X, _ = _make_synthetic_data(T=200, k=2)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)
        assert np.isfinite(result.log_likelihood)

    def test_filtered_se_positive(self):
        y, X, _ = _make_synthetic_data(T=200, k=2)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)
        assert (result.filtered_se >= 0).all().all()
        assert (result.smoothed_se >= 0).all().all()

    def test_betas_nonzero(self):
        """Regression test: filtered betas must NOT be near-zero."""
        y, X, _ = _make_synthetic_data(T=500, k=3, seed=42)
        kf = KalmanFilter(maxiter=100)
        result = kf.filter(y, X)

        mean_abs_beta = result.filtered_betas.iloc[100:].abs().mean().mean()
        assert mean_abs_beta > 0.1, (
            f"Betas near zero: mean |beta| = {mean_abs_beta:.4f}"
        )

    def test_Q_nonzero(self):
        """Regression test: Q must not collapse to zero."""
        y, X, _ = _make_synthetic_data(T=500, k=2)
        kf = KalmanFilter(maxiter=100)
        result = kf.filter(y, X)
        q_diag = np.diag(result.Q)
        assert np.all(q_diag > 1e-12), f"Q collapsed to zero: {q_diag}"

    def test_tracks_true_betas(self):
        """Filtered betas should be close to true betas."""
        y, X, beta_true = _make_synthetic_data(T=500, k=2, seed=123)
        kf = KalmanFilter(maxiter=100)
        result = kf.filter(y, X)

        burn = 100
        for j in range(2):
            filtered = result.filtered_betas.iloc[burn:, j].to_numpy()
            true = beta_true[burn:, j]
            mae = np.mean(np.abs(filtered - true))
            assert mae < 0.2, f"Factor {j}: MAE={mae:.3f} too high"

    def test_single_factor(self):
        y, X, _ = _make_synthetic_data(T=200, k=1)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)
        assert result.filtered_betas.shape == (200, 1)
        assert result.Q.shape == (1, 1)

    def test_fallback_to_ols(self):
        """OLS fallback should produce valid output."""
        y, X, _ = _make_synthetic_data(T=50, k=2)
        kf = KalmanFilter(maxiter=50, fallback_to_ols=True)
        result = kf.filter(y, X)
        assert isinstance(result, KalmanResult)
        assert np.isfinite(result.log_likelihood)
        assert result.filtered_se.shape == (50, 2)

    def test_train_end_idx(self):
        """Test train/apply split via train_end_idx."""
        y, X, _ = _make_synthetic_data(T=300, k=2)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X, train_end_idx=200)

        assert isinstance(result, KalmanResult)
        assert result.filtered_betas.shape == (300, 2)
        assert result.filtered_se.shape == (300, 2)
        assert np.isfinite(result.log_likelihood)

    def test_train_end_idx_same_as_full(self):
        """train_end_idx=T should behave like no split."""
        y, X, _ = _make_synthetic_data(T=200, k=2, seed=99)
        kf = KalmanFilter(maxiter=50)
        result_full = kf.filter(y, X)
        result_split = kf.filter(y, X, train_end_idx=200)
        # Both should produce finite results
        assert np.isfinite(result_full.log_likelihood)
        assert np.isfinite(result_split.log_likelihood)


class TestKalmanResultProperties:

    def test_frozen(self):
        y, X, _ = _make_synthetic_data(T=100, k=2)
        kf = KalmanFilter(maxiter=50)
        result = kf.filter(y, X)
        with pytest.raises(AttributeError):
            result.R = 999.0
