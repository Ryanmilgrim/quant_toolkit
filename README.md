# quant_toolkit

A pure-Python library for quantitative portfolio analysis. It provides Fama-French
data access, benchmark style decomposition, PCA + GARCH factor risk models, Markov
Switching regime detection, and Black-Scholes option pricing -- all without web
framework dependencies.

## Capabilities

### 01 -- Data: Fama-French Industry Portfolios & Factors

Daily and monthly value-weighted industry returns from Kenneth French's Data Library,
plus factor sets (FF3, FF5, FF5+Mom). Log returns by default for compounding consistency.

```python
from datetime import date
from toolkit import fetch_ff_industry_daily, fetch_ff_factors_daily

industries = fetch_ff_industry_daily(10, start_date=date(2020, 1, 1))
factors = fetch_ff_factors_daily(factor_set="ff3", start_date=date(2020, 1, 1))
```

### 02 -- Style Analysis: Rolling Benchmark Replication

Constrained least-squares decomposition of a benchmark into a long-only mix of
industry returns. Rolling window re-estimation with projection or QP solver.

```python
from toolkit import get_universe_returns, StyleAnalysis
import matplotlib.pyplot as plt

uni = get_universe_returns(10, start_date=date(2000, 1, 1))
sa = StyleAnalysis(uni, benchmark_name="Mkt")
run = sa.run(style_window_years=1.0, optimize_frequency="monthly")

print(run.summary())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
run.plot_growth(ax=axes[0, 0])
run.plot_tracking_error(ax=axes[0, 1])
run.plot_weights(ax=axes[1, 0])
run.plot_weights_snapshot(ax=axes[1, 1])
fig.tight_layout()
plt.show()
```

### 03 -- Risk Model: PCA Factor Rotation with GARCH(1,1)

LassoCV selects factor exposures, PCA rotates into principal components, and
GARCH(1,1) forecasts time-varying conditional variance. Kalman-filtered betas
capture time-varying factor loadings. Train/test evaluation of volatility and
correlation forecasts.

```python
from toolkit import RiskModel

rm = RiskModel(rf_name="Rf", garch_dist="t", pca_demean=False)
run = rm.evaluate_train_test(
    assets=assets, factors=factors, rf=rf,
    train_fraction=0.7, realized_window=60, progress=True,
)
print(run.summary())

# Beta loadings, covariance, evaluation metrics
run.beta_loadings
cov = run.asset_cov_at(date)
ev = run["evaluation"]
print(ev["summary"])
```

### 04 -- Regime Detection: Markov Switching Models

Hidden Markov Models (Hamilton, 1989) with switching mean, variance, and optional
autoregressive terms. Produces soft regime probabilities for economic series from
FRED. Iteratively build a signal library (RegimeCollection) for downstream ML.

```python
from toolkit.analysis.regime_detection import RegimeConfig, RegimeDetector

config = RegimeConfig(k_regimes=2, switching_variance=True, switching_trend=True)
detector = RegimeDetector(config)
result = detector.fit(series)
print(result.regime_summary())
```

### 05 -- Options: Black-Scholes European Pricing

Closed-form pricing for European calls and puts with continuous compounding.

```python
from toolkit import black_scholes_price

call = black_scholes_price(
    spot=100, strike=105, time_to_expiry=0.5,
    risk_free_rate=0.05, volatility=0.20, option_type="call",
)
put = black_scholes_price(
    spot=100, strike=105, time_to_expiry=0.5,
    risk_free_rate=0.05, volatility=0.20, option_type="put",
)
print(f"Call price: ${call:.4f}")
print(f"Put price:  ${put:.4f}")
```

### Plotly.js payloads (for web apps)

```python
from toolkit.plotly_payload import summarize_style_run

summary = summarize_style_run(run)
chart_data = summary["chart_growth"]   # {series: [{name, x, y}], y_axis_title}
```

## Installation

```bash
pip install .
# or for development:
pip install -e ".[dev]"
```

## Example scripts

Runnable demos are in the `examples/` directory:

| Script | Description |
|--------|-------------|
| `examples/black_scholes_pricing.py` | Price European call and put options |
| `examples/fama_french_data.py` | Fetch Fama-French industry and factor data |
| `examples/style_analysis.py` | Rolling style decomposition with charts |
| `examples/factor_analysis.py` | PCA + GARCH factor risk model with evaluation |
| `examples/regime_detection.py` | Markov Switching regime signal building workflow |
| `examples/chart_gallery.py` | All matplotlib chart functions for style analysis |

```bash
python examples/black_scholes_pricing.py
python examples/factor_analysis.py
python examples/regime_detection.py
```

## Project structure

```
toolkit/
  __init__.py            # Top-level public API
  charts.py              # Matplotlib chart functions (style analysis)
  factor_charts.py       # Matplotlib chart functions (factor/risk model)
  regime_charts.py       # Matplotlib chart functions (regime detection)
  plotly_payload.py      # Plotly.js payload builders (web bridge)
  returns.py             # Return transforms (simple <-> log)
  analysis/              # Analytical models
    benchmark_style.py   # StyleAnalysis, StyleRun
    risk_model.py        # RiskModel, RiskModelRun (PCA + GARCH)
    factor_analysis.py   # Legacy aliases (FactorModel -> RiskModel)
    kalman_filter.py     # Kalman filter for time-varying betas
    black_scholes.py     # European option pricing
    regime_detection.py  # RegimeDetector, RegimeConfig, RegimeCollection
    transforms.py        # Stationarity transforms for time series
    style_storage.py     # Save/load style analysis snapshots
    risk_storage.py      # Save/load risk model snapshots
    factor_storage.py    # Legacy aliases for risk storage
    regime_storage.py    # Save/load regime collections
  data/                  # Market data adapters
    french_industry.py   # Fama-French Data Library fetcher
  universe/              # Universe construction
    loader.py            # get_universe_returns(), get_universe_start_date()
tests/                   # Unit tests
examples/                # Runnable example scripts
pyproject.toml           # Package metadata and dependencies
```

## Dependencies

- **Runtime**: pandas, numpy, requests, matplotlib, scikit-learn, statsmodels, arch, tqdm
- **Dev**: pytest

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Companion web dashboard

See [quant_website](https://github.com/Ryanmilgrim/quant_website) for a Flask
dashboard that uses this toolkit to visualise analyses interactively with Plotly.js.
