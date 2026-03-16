"""FRED (Federal Reserve Economic Data) API client.

Fetches economic time series from the St. Louis Fed's FRED API.
Follows the same caching pattern as :mod:`toolkit.data.french_industry`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

FRED_API_BASE = "https://api.stlouisfed.org/fred"


@dataclass(frozen=True)
class FredConfig:
    """Configuration for FRED API access."""

    api_key: str = field(
        default_factory=lambda: os.environ.get("FRED_API_KEY", "fff3aac29f57163f8e8b3ed5ccb55fd9")
    )
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "fred"
    )
    timeout_s: float = 30.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "FRED API key is required. Set the FRED_API_KEY environment "
                "variable or pass api_key to FredConfig."
            )


def _fred_get(
    endpoint: str,
    params: dict,
    cfg: FredConfig,
) -> dict:
    """Make a GET request to the FRED API with retries."""
    url = f"{FRED_API_BASE}/{endpoint}"
    params = {**params, "api_key": cfg.api_key, "file_type": "json"}

    last_exc: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=cfg.timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt < cfg.max_retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"FRED API request to {endpoint} failed after {cfg.max_retries} "
        f"attempts"
    ) from last_exc


def _cache_path(cfg: FredConfig, series_id: str, start: str, end: str) -> Path:
    """Build the cache file path for a series request."""
    safe_id = series_id.upper().replace("/", "_")
    filename = f"{safe_id}_{start}_{end}.json"
    return cfg.cache_dir / filename


def fetch_fred_series(
    series_id: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cfg: Optional[FredConfig] = None,
    refresh: bool = False,
) -> pd.Series:
    """Fetch a time series from FRED.

    Parameters
    ----------
    series_id
        FRED series identifier (e.g. ``"GDP"``, ``"FEDFUNDS"``, ``"T10Y2Y"``).
    start
        Start date as ISO string (e.g. ``"1960-01-01"``). Defaults to earliest.
    end
        End date as ISO string. Defaults to latest available.
    cfg
        FRED API configuration. Built from environment if ``None``.
    refresh
        If ``True``, bypass the local cache.

    Returns
    -------
    pd.Series
        Time series with DatetimeIndex and ``name`` set to *series_id*.
    """
    if cfg is None:
        cfg = FredConfig()

    obs_start = start or "1776-07-04"
    obs_end = end or "9999-12-31"

    # Check cache
    cache_file = _cache_path(cfg, series_id, obs_start, obs_end)
    if cache_file.exists() and not refresh:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        params = {
            "series_id": series_id,
            "observation_start": obs_start,
            "observation_end": obs_end,
        }
        data = _fred_get("series/observations", params, cfg)

        # Cache to disk
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    # Parse observations
    observations = data.get("observations", [])
    if not observations:
        raise ValueError(f"No observations returned for FRED series '{series_id}'")

    dates = []
    values = []
    for obs in observations:
        val = obs.get("value", ".")
        if val == "." or val == "":
            continue
        try:
            values.append(float(val))
            dates.append(pd.Timestamp(obs["date"]))
        except (ValueError, TypeError):
            continue

    if not values:
        raise ValueError(f"No valid observations for FRED series '{series_id}'")

    return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)


def search_fred_series(
    query: str,
    *,
    limit: int = 10,
    cfg: Optional[FredConfig] = None,
) -> list[dict]:
    """Search FRED for series matching a query.

    Returns a list of dicts with keys: ``id``, ``title``, ``frequency``,
    ``units``, ``seasonal_adjustment``.
    """
    if cfg is None:
        cfg = FredConfig()

    data = _fred_get(
        "series/search",
        {"search_text": query, "limit": limit, "order_by": "popularity"},
        cfg,
    )

    results = []
    for s in data.get("seriess", []):
        results.append({
            "id": s.get("id", ""),
            "title": s.get("title", ""),
            "frequency": s.get("frequency_short", ""),
            "units": s.get("units_short", ""),
            "seasonal_adjustment": s.get("seasonal_adjustment_short", ""),
        })
    return results


def get_fred_series_info(
    series_id: str,
    *,
    cfg: Optional[FredConfig] = None,
) -> dict:
    """Fetch metadata for a single FRED series."""
    if cfg is None:
        cfg = FredConfig()

    data = _fred_get("series", {"series_id": series_id}, cfg)
    seriess = data.get("seriess", [])
    if not seriess:
        raise ValueError(f"FRED series '{series_id}' not found")
    s = seriess[0]
    return {
        "id": s.get("id", ""),
        "title": s.get("title", ""),
        "frequency": s.get("frequency_short", ""),
        "units": s.get("units_short", ""),
        "seasonal_adjustment": s.get("seasonal_adjustment_short", ""),
        "observation_start": s.get("observation_start", ""),
        "observation_end": s.get("observation_end", ""),
        "notes": s.get("notes", ""),
    }


__all__ = [
    "FredConfig",
    "fetch_fred_series",
    "get_fred_series_info",
    "search_fred_series",
]
