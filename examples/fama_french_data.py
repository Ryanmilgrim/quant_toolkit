"""Fama-French Data -- example usage.

Demonstrates:
  - Fetching daily industry returns at multiple granularities
  - Fetching factor data (FF3, FF5)
  - Constructing a universe via get_universe_returns()

Run with::

    python examples/fama_french_data.py
"""

from datetime import date

from toolkit.data import fetch_ff_industry_daily, fetch_ff_factors_daily
from toolkit.universe import get_universe_returns

# Fetch 10-industry daily returns (value-weighted, log returns)
industries = fetch_ff_industry_daily(10, weighting="value", start_date=date(2020, 1, 1))
print("=== 10-Industry Daily Returns (first 5 rows) ===")
print(industries.head())
print(f"Shape: {industries.shape}")
print()

# Fetch 3-factor model data
factors = fetch_ff_factors_daily(factor_set="ff3", start_date=date(2020, 1, 1))
print("=== FF 3-Factor Daily Returns (first 5 rows) ===")
print(factors.head())
print()

# Fetch a combined universe (assets + factors + benchmarks)
uni = get_universe_returns(10, start_date=date(2020, 1, 1))
print("=== Combined Universe ===")
print(f"Groups: {uni.columns.get_level_values(0).unique().tolist()}")
print(f"Shape:  {uni.shape}")
print(uni.head())
