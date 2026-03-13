# quant_toolkit

Pure-Python quantitative finance toolkit for Fama-French data, benchmark style analysis, factor risk models, Black-Scholes pricing, and matplotlib charting.

## Installation

```bash
pip install .
# or for development:
pip install -e ".[dev]"
```

## Usage

### Black-Scholes option pricing

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

Output:

```
Spot: $100.00  Strike: $105.00
Time: 0.5 yr  Rate: 5.0%  Vol: 20.0%
Call price: $4.5817
Put price:  $6.9892
```

### Fama-French industry data

```python
from datetime import date
from toolkit import fetch_ff_industry_daily, fetch_ff_factors_daily

# 10-industry daily returns (log, value-weighted)
industries = fetch_ff_industry_daily(10, start_date=date(2020, 1, 1))

# 3-factor model
factors = fetch_ff_factors_daily(factor_set="ff3", start_date=date(2020, 1, 1))
```

### Benchmark style analysis

```python
from toolkit import get_universe_returns, StyleAnalysis
import matplotlib.pyplot as plt

uni = get_universe_returns(10, start_date=date(2000, 1, 1))
sa = StyleAnalysis(uni, benchmark_name="Mkt")
run = sa.run(style_window_years=1.0, optimize_frequency="monthly")

print(run.summary())

# Built-in matplotlib charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
run.plot_growth(ax=axes[0, 0])
run.plot_tracking_error(ax=axes[0, 1])
run.plot_weights(ax=axes[1, 0])
run.plot_weights_snapshot(ax=axes[1, 1])
fig.tight_layout()
plt.show()
```

Output:

```
StyleRun
  benchmark: Mkt
  window:    12 (1.00 yrs, monthly)
  rebalance: monthly
  method:    projection
  assets:    12
  ...

portfolio:
  total_return: 11.8561
  ann_return: 2.9035
  ann_vol: 1.9540
  sharpe: 1.4859

active:
  total_return: -5.4076
  ann_return: -1.3243
  ann_vol: 0.9490
  info_ratio: -1.3955
```

### Factor risk model (PCA + GARCH)

```python
from toolkit import FactorModel

fm = FactorModel(rf_name="Rf", garch_dist="t", pca_demean=False)
run = fm.evaluate_train_test(
    assets=assets, factors=factors, rf=rf,
    train_fraction=0.7, realized_window=60, progress=True,
)
print(run.summary())
```

Output:

```
FactorRun
  as_of:    2017-09-06
  train_end:2016-11-17
  n_obs:    700
  n_train:  490
  n_assets: 25
  n_factors:5
  n_pcs:    5
```

Access the fitted model outputs:

```python
# Beta loadings (n_assets x n_factors)
run.beta_loadings

# Time-varying covariance matrix at a specific date
cov = run.asset_cov_at(date)

# Forecast covariance (latest date)
run["asset_cov_forecast"]

# Train/test evaluation metrics
ev = run["evaluation"]
print(ev["summary"])
```

Output:

```
vol_mse_in_sample:           mean=1.49e-06  median=1.44e-06
vol_mse_out_of_sample:       mean=2.27e-06  median=2.19e-06
corr_frob_err_in_sample:     mean=2.92      median=2.87
corr_frob_err_out_of_sample: mean=3.20      median=3.17
```

Built-in factor charts:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
run.plot_beta_heatmap(ax=axes[0, 0])
run.plot_factor_risk_heatmap(metric="correlation", ax=axes[0, 1])
run.plot_volatility_backtest(asset="A1", ax=axes[1, 0])
run.plot_returns_with_confidence_bands(asset="A1", ax=axes[1, 1])
fig.tight_layout()
plt.show()
```

### Plotly.js payloads (for web apps)

```python
from toolkit.plotly_payload import summarize_style_run

# Returns JSON-serialisable dicts for Plotly.js rendering
summary = summarize_style_run(run)
chart_data = summary["chart_growth"]   # {series: [{name, x, y}], y_axis_title}
```

## Example scripts

Runnable demos are in the `examples/` directory:

| Script | Description |
|--------|-------------|
| `examples/black_scholes_pricing.py` | Price European call and put options |
| `examples/fama_french_data.py` | Fetch Fama-French industry and factor data |
| `examples/style_analysis.py` | Run style analysis on synthetic data with charts |
| `examples/factor_analysis.py` | Factor risk model with evaluation and charts |
| `examples/chart_gallery.py` | Demonstrate all style analysis chart functions |

```bash
python examples/black_scholes_pricing.py
python examples/factor_analysis.py
```

## Project structure

```
toolkit/
  __init__.py            # Top-level public API
  charts.py              # Matplotlib chart functions (style analysis)
  factor_charts.py       # Matplotlib chart functions (factor analysis)
  plotly_payload.py      # Plotly.js payload builders (web bridge)
  returns.py             # Return transforms (simple <-> log)
  analysis/              # Analytical models
    benchmark_style.py   # StyleAnalysis, StyleRun
    factor_analysis.py   # FactorModel, FactorRun (PCA + GARCH risk model)
    black_scholes.py     # European option pricing
    style_storage.py     # Save/load analysis snapshots
  data/                  # Market data adapters
    french_industry.py   # Fama-French Data Library fetcher
  universe/              # Universe construction
    loader.py            # get_universe_returns(), get_universe_start_date()
tests/                   # Unit tests (58 tests)
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

```
========================= 58 passed =========================
```

## Companion web dashboard

See [quant_website](https://github.com/Ryanmilgrim/quant_website) for a Flask dashboard that uses this toolkit to visualise analyses interactively with Plotly.js.
