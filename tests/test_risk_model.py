import numpy as np
import pandas as pd
import pytest

from toolkit.analysis.risk_model import (
    RiskModel,
    RiskModelRun,
    FactorModel,
    FactorRun,
    annualize_vol,
    covariance_to_correlation,
    covariance_to_correlation_df,
)


def _synthetic_factor_universe(
    *,
    periods: int = 500,
    n_assets: int = 10,
    n_factors: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build synthetic (assets, factors, rf) aligned DataFrames."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=periods)

    factors = pd.DataFrame(
        rng.normal(0, 0.007, size=(periods, n_factors)),
        index=dates,
        columns=[f"F{i + 1}" for i in range(n_factors)],
    )
    B_true = rng.normal(0, 0.8, size=(n_assets, n_factors))
    resid = rng.normal(0, 0.01, size=(periods, n_assets))

    assets_excess = factors.to_numpy() @ B_true.T + resid
    rf = pd.Series(
        rng.normal(0.00005, 0.0001, size=periods),
        index=dates, name="Rf",
    )
    assets = pd.DataFrame(
        assets_excess, index=dates,
        columns=[f"A{i + 1}" for i in range(n_assets)],
    ) + rf.values.reshape(-1, 1)

    return assets, factors, rf


def _synthetic_factor_uni(
    *,
    periods: int = 500,
    n_assets: int = 10,
    n_factors: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a MultiIndex DataFrame matching get_universe_returns format."""
    assets, factors, rf = _synthetic_factor_universe(
        periods=periods, n_assets=n_assets, n_factors=n_factors, seed=seed,
    )
    benchmarks = rf.to_frame()
    return pd.concat(
        [assets, factors, benchmarks],
        axis=1,
        keys=["assets", "factors", "benchmarks"],
    )


# ---------------------------------------------------------------------------
# Standalone helper tests
# ---------------------------------------------------------------------------

def test_annualize_vol_scalar_series() -> None:
    s = pd.Series([0.01, 0.02, 0.03])
    result = annualize_vol(s, trading_days=252)
    expected = s * np.sqrt(252)
    pd.testing.assert_series_equal(result, expected)


def test_covariance_to_correlation_identity() -> None:
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    corr = covariance_to_correlation(cov)
    np.testing.assert_allclose(corr, cov, atol=1e-12)


def test_covariance_to_correlation_diagonal_is_one() -> None:
    rng = np.random.default_rng(99)
    A = rng.normal(size=(50, 4))
    cov = A.T @ A / 50
    corr = covariance_to_correlation(cov, clip=True)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)
    assert (corr >= -1.0 - 1e-12).all()
    assert (corr <= 1.0 + 1e-12).all()


def test_covariance_to_correlation_rejects_nonsquare() -> None:
    with pytest.raises(ValueError, match="square"):
        covariance_to_correlation(np.zeros((2, 3)))


def test_covariance_to_correlation_df_preserves_labels() -> None:
    idx = ["a", "b"]
    cov = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=idx, columns=idx)
    corr = covariance_to_correlation_df(cov)
    assert list(corr.index) == idx
    assert list(corr.columns) == idx


# ---------------------------------------------------------------------------
# FactorRun tests (mock data, no sklearn needed)
# ---------------------------------------------------------------------------

def _mock_factor_run() -> FactorRun:
    """Build a minimal FactorRun with mock results for property tests."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    n_assets, n_factors, n_pcs = 3, 2, 2
    asset_names = ["A1", "A2", "A3"]
    factor_names = ["F1", "F2"]
    pc_names = ["PC1", "PC2"]

    beta = pd.DataFrame(
        np.eye(n_assets, n_factors), index=asset_names, columns=factor_names,
    )
    ev = pd.DataFrame(
        np.eye(n_factors, n_pcs), index=factor_names, columns=pc_names,
    )
    pc_cv = pd.DataFrame(
        np.full((10, n_pcs), 0.0001), index=dates, columns=pc_names,
    )
    resid_cv = pd.DataFrame(
        np.full((10, n_assets), 0.0001), index=dates, columns=asset_names,
    )
    pc_fv = pc_cv.copy()
    resid_fv = resid_cv.copy()

    results = {
        "meta": {"as_of_date": dates[-1], "train_end": None,
                 "n_obs": 10, "n_train": 10, "n_assets": 3,
                 "n_factors": 2, "n_pcs": 2},
        "beta_loadings": beta,
        "eigen_vectors": ev,
        "pc_cond_var": pc_cv,
        "pc_forecast_var": pc_fv,
        "resid_cond_var": resid_cv,
        "resid_forecast_var": resid_fv,
    }
    return FactorRun(params={}, results=results)


def test_factor_run_getitem() -> None:
    run = _mock_factor_run()
    assert isinstance(run["beta_loadings"], pd.DataFrame)


def test_factor_run_properties() -> None:
    run = _mock_factor_run()
    assert run.assets == ["A1", "A2", "A3"]
    assert run.factors == ["F1", "F2"]
    assert run.beta_loadings.shape == (3, 2)
    assert run.eigen_vectors.shape == (2, 2)


def test_factor_cov_at_shape() -> None:
    run = _mock_factor_run()
    dt = run["pc_cond_var"].index[0]
    cov = run.factor_cov_at(dt)
    assert cov.shape == (2, 2)
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-12)


def test_factor_corr_at_diagonal() -> None:
    run = _mock_factor_run()
    dt = run["pc_cond_var"].index[0]
    corr = run.factor_corr_at(dt)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-12)


def test_asset_cov_at_shape() -> None:
    run = _mock_factor_run()
    dt = run["pc_cond_var"].index[0]
    cov = run.asset_cov_at(dt)
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-12)


def test_asset_cov_at_positive_semidefinite() -> None:
    run = _mock_factor_run()
    dt = run["pc_cond_var"].index[0]
    cov = run.asset_cov_at(dt)
    eigvals = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigvals >= -1e-10).all()


def test_iter_asset_covariances() -> None:
    run = _mock_factor_run()
    pairs = list(run.iter_asset_covariances())
    assert len(pairs) == 10
    for dt, cov in pairs:
        assert isinstance(dt, pd.Timestamp)
        assert cov.shape == (3, 3)


def test_factor_run_summary() -> None:
    run = _mock_factor_run()
    s = run.summary()
    assert "RiskModelRun" in s
    assert "n_assets" in s


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------

def test_factor_model_run_outputs_well_formed() -> None:
    assets, factors, rf = _synthetic_factor_universe(
        periods=300, n_assets=5, n_factors=3, seed=1,
    )
    fm = FactorModel(rf_name="Rf", garch_dist="t", pca_demean=False)
    run = fm.run(assets=assets, factors=factors, rf=rf, progress=False)

    assert isinstance(run, FactorRun)
    assert run.beta_loadings.shape == (5, 3)
    assert run.eigen_vectors.shape[0] == 3
    assert run["asset_cov_forecast"].shape == (5, 5)
    assert np.isfinite(run.beta_loadings.to_numpy()).all()


def test_factor_model_with_uni_input() -> None:
    uni = _synthetic_factor_uni(periods=300, n_assets=5, n_factors=3, seed=2)
    fm = FactorModel(rf_name="Rf")
    run = fm.run(uni=uni, progress=False)

    assert isinstance(run, FactorRun)
    assert run["beta_loadings"].shape[0] == 5


def test_factor_model_evaluate_train_test() -> None:
    assets, factors, rf = _synthetic_factor_universe(
        periods=400, n_assets=5, n_factors=3, seed=3,
    )
    fm = FactorModel(rf_name="Rf", garch_dist="t")
    run = fm.evaluate_train_test(
        assets=assets, factors=factors, rf=rf,
        train_fraction=0.7, realized_window=60, progress=False,
    )

    assert "evaluation" in run.results
    ev = run["evaluation"]
    assert "summary" in ev
    assert "timeseries" in ev
    assert "pred_vol" in ev["timeseries"]
    assert "real_vol" in ev["timeseries"]


def test_covariance_symmetric_psd() -> None:
    assets, factors, rf = _synthetic_factor_universe(
        periods=300, n_assets=5, n_factors=3, seed=4,
    )
    fm = FactorModel(rf_name="Rf")
    run = fm.run(assets=assets, factors=factors, rf=rf, progress=False)

    last_dt = run["asset_cond_var"].index[-1]
    cov = run.asset_cov_at(last_dt)
    cov_arr = cov.to_numpy()
    np.testing.assert_allclose(cov_arr, cov_arr.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(cov_arr)
    assert (eigvals >= -1e-10).all()


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

def test_factor_model_rejects_bad_uni() -> None:
    bad = pd.DataFrame({"a": [1, 2, 3]})
    fm = FactorModel()
    with pytest.raises(TypeError, match="MultiIndex"):
        fm.run(uni=bad)


def test_factor_model_rejects_missing_inputs() -> None:
    fm = FactorModel()
    with pytest.raises(ValueError, match="Provide either"):
        fm.run()


def test_factor_model_rejects_short_training() -> None:
    assets, factors, rf = _synthetic_factor_universe(
        periods=100, n_assets=3, n_factors=2, seed=5,
    )
    fm = FactorModel()
    with pytest.raises(ValueError, match="too short"):
        fm._fit_train_and_filter_full(
            assets=assets, factors=factors, rf=rf,
            train_end=pd.Timestamp(assets.index[10]),
            progress=False,
        )


# ---------------------------------------------------------------------------
# Kalman filter integration tests
# ---------------------------------------------------------------------------

def test_pipeline_produces_kalman_results() -> None:
    """Verify all new Kalman-related keys exist in results."""
    assets, factors, rf = _synthetic_factor_universe(
        periods=300, n_assets=5, n_factors=3, seed=10,
    )
    fm = FactorModel(rf_name="Rf", garch_dist="t", pca_demean=False)
    run = fm.run(assets=assets, factors=factors, rf=rf, progress=False)

    assert "beta_ts" in run.results
    assert "alpha_ts" in run.results
    assert "beta_se" in run.results
    assert "kalman_diagnostics" in run.results

    # Check that each asset has entries
    for a in run.assets:
        assert a in run.results["beta_ts"]
        assert a in run.results["alpha_ts"]
        assert a in run.results["beta_se"]
        assert a in run.results["kalman_diagnostics"]
        diag = run.results["kalman_diagnostics"][a]
        assert "R" in diag
        assert "Q" in diag
        assert "log_likelihood" in diag


def test_kalman_residuals_shape() -> None:
    """Residuals should have same shape as assets."""
    assets, factors, rf = _synthetic_factor_universe(
        periods=300, n_assets=5, n_factors=3, seed=11,
    )
    fm = FactorModel(rf_name="Rf")
    run = fm.run(assets=assets, factors=factors, rf=rf, progress=False)

    resid = run["resid"]
    assert resid.shape[1] == 5
    assert resid.shape[0] == run.meta["n_obs"]


def test_beta_matrix_at() -> None:
    """beta_matrix_at should return correct shape."""
    assets, factors, rf = _synthetic_factor_universe(
        periods=300, n_assets=5, n_factors=3, seed=12,
    )
    fm = FactorModel(rf_name="Rf")
    run = fm.run(assets=assets, factors=factors, rf=rf, progress=False)

    dt = run["resid"].index[150]
    bmat = run.beta_matrix_at(dt)
    assert bmat.shape == (5, 3)
    assert list(bmat.index) == run.assets
    assert list(bmat.columns) == run.factors


def test_asset_cov_still_psd() -> None:
    """Covariance matrices must remain PSD with Kalman betas."""
    assets, factors, rf = _synthetic_factor_universe(
        periods=300, n_assets=5, n_factors=3, seed=13,
    )
    fm = FactorModel(rf_name="Rf")
    run = fm.run(assets=assets, factors=factors, rf=rf, progress=False)

    last_dt = run["asset_cond_var"].index[-1]
    cov = run.asset_cov_at(last_dt)
    cov_arr = cov.to_numpy()
    np.testing.assert_allclose(cov_arr, cov_arr.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(cov_arr)
    assert (eigvals >= -1e-10).all()
