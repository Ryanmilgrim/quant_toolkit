# quant_toolkit — Agent Instructions (AGENTS.md)

## What this repo is
A reusable **pure-Python quantitative finance toolkit** (no web framework dependencies).

This is a library-only package. The companion Flask web dashboard lives in a
separate repo: [quant_website](https://github.com/Ryanmilgrim/quant_website).

## Architecture

### Package layout

```
toolkit/
  __init__.py            # Top-level public API
  charts.py              # Matplotlib chart functions (plot_growth, plot_tracking_error, etc.)
  plotly_payload.py      # Plotly.js JSON payload builders (bridge for web apps)
  returns.py             # Return transforms (simple <-> log)
  analysis/              # Analytical models
    benchmark_style.py   # StyleAnalysis, StyleRun (with plot methods)
    black_scholes.py     # European option pricing
    style_storage.py     # Save/load analysis snapshots (pickle)
  data/                  # Market data adapters
    french_industry.py   # Fama-French Data Library fetcher
  universe/              # Universe construction
    loader.py            # get_universe_returns(), get_universe_start_date()
tests/                   # Unit tests
examples/                # Runnable example scripts
pyproject.toml           # Package metadata and dependencies
```

### Hard rules
- `toolkit/**` **must not import Flask** or any web framework.
- All IO (network, file) is isolated in `data/` adapters and `analysis/style_storage.py`.
- Pricing/risk/math functions must be deterministic and side-effect free.

## Development workflow

### Setup
```bash
pip install -e ".[dev]"
```

### After any code change (required)
```bash
python -m compileall toolkit/
pytest
```

### Keep diffs small and reviewable
- Prefer small, cohesive commits.
- Avoid drive-by reformatting.

## Design standards

### API & stability
- Small, well-named functions/classes with type hints.
- Chart functions follow the matplotlib `ax` pattern (`ax=None` creates a new figure).
- `StyleRun` has convenience plot methods that delegate to `charts.py`.

### Numerics & correctness
- Validate inputs explicitly (non-negative vol, positive time to expiry).
- Handle edge cases (empty DataFrames, NaN weights, zero vol).
- Tests must be deterministic and run without network access.

## Dependencies
- Runtime: pandas, numpy, requests, matplotlib
- Dev: pytest
- No Flask, no yfinance

## Documentation
- Update README when capabilities change.
- Add example scripts for new features in `examples/`.
