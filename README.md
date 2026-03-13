# quant_toolkit

Pure-Python quantitative finance toolkit for Fama-French data, benchmark style analysis, Black-Scholes pricing, and matplotlib charting.

## Installation

```bash
pip install .
# or for development:
pip install -e ".[dev]"
```

## Usage

### Black-Scholes option pricing

```python
from quant_toolkit import black_scholes_price

call = black_scholes_price(
    spot=100, strike=105, time_to_expiry=0.5,
    risk_free_rate=0.05, volatility=0.20, option_type="call",
)
print(f"Call price: ${call:.2f}")
```

### Fama-French industry data

```python
from datetime import date
from quant_toolkit import fetch_ff_industry_daily, fetch_ff_factors_daily

# 10-industry daily returns (log, value-weighted)
industries = fetch_ff_industry_daily(10, start_date=date(2020, 1, 1))

# 3-factor model
factors = fetch_ff_factors_daily(factor_set="ff3", start_date=date(2020, 1, 1))
```

### Benchmark style analysis with charts

```python
from quant_toolkit import get_universe_returns, StyleAnalysis
import matplotlib.pyplot as plt

uni = get_universe_returns(10, start_date=date(2000, 1, 1))
sa = StyleAnalysis(uni, benchmark_name="Mkt")
run = sa.run(style_window_years=1.0, optimize_frequency="monthly")

# Built-in matplotlib charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
run.plot_growth(ax=axes[0, 0])
run.plot_tracking_error(ax=axes[0, 1])
run.plot_weights(ax=axes[1, 0])
run.plot_weights_snapshot(ax=axes[1, 1])
fig.tight_layout()
plt.show()
```

### Performance metrics

```python
perf = run.performance()
print(perf["portfolio"])   # total_return, ann_return, ann_vol, sharpe
print(perf["active"])      # includes info_ratio
```

### Plotly.js payloads (for web apps)

```python
from quant_toolkit.plotly_payload import summarize_style_run

# Returns JSON-serialisable dicts for Plotly.js rendering
summary = summarize_style_run(run)
chart_data = summary["chart_growth"]   # {series: [{name, x, y}], y_axis_title}
```

## Project structure

```
quant_toolkit/
  __init__.py            # Top-level public API
  charts.py              # Matplotlib chart functions
  plotly_payload.py      # Plotly.js payload builders (web bridge)
  returns.py             # Return transforms (simple <-> log)
  analysis/              # Analytical models
    benchmark_style.py   # StyleAnalysis, StyleRun
    black_scholes.py     # European option pricing
    style_storage.py     # Save/load analysis snapshots
  data/                  # Market data adapters
    french_industry.py   # Fama-French Data Library fetcher
  universe/              # Universe construction
    loader.py            # get_universe_returns(), get_universe_start_date()
tests/                   # Unit tests (26 tests)
examples/                # Runnable example scripts
pyproject.toml           # Package metadata and dependencies
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Companion web dashboard

See [quant_website](https://github.com/Ryanmilgrim/quant_website) for a Flask dashboard that uses this toolkit to visualise analyses interactively with Plotly.js.
