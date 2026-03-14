from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import pickle
from typing import Optional

import pandas as pd

from .factor_analysis import FactorRun
from .style_storage import normalize_snapshot_name, snapshot_path


@dataclass
class FactorAnalysisSnapshot:
    """Persisted factor analysis snapshot."""

    name: str
    created_at: datetime
    universe: int
    weighting: str
    factor_set: str
    start_date: Optional[date]
    end_date: Optional[date]
    garch_dist: str
    pca_demean: bool
    train_fraction: float
    realized_window: int
    run: FactorRun
    universe_data: pd.DataFrame

    @property
    def key(self) -> str:
        return normalize_snapshot_name(self.name)


@dataclass(frozen=True)
class FactorSnapshotInfo:
    """Lightweight metadata for listing saved factor snapshots."""

    key: str
    name: str
    created_at: datetime
    universe: int
    weighting: str
    factor_set: str
    start_date: Optional[date]
    end_date: Optional[date]
    garch_dist: str
    pca_demean: bool
    train_fraction: float
    realized_window: int
    n_assets: int
    n_factors: int
    n_pcs: int
    as_of_date: Optional[date]
    train_end: Optional[date]


def save_factor_snapshot(
    snapshot: FactorAnalysisSnapshot,
    directory: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a factor snapshot to a single pickle file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(directory, snapshot.name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Snapshot already exists: {snapshot.name}")
    with path.open("wb") as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_factor_snapshot(path: Path) -> FactorAnalysisSnapshot:
    """Load a factor snapshot from a pickle file."""
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, FactorAnalysisSnapshot):
        raise TypeError("Invalid snapshot payload.")
    return obj


def list_factor_snapshots(directory: Path) -> list[FactorSnapshotInfo]:
    """List saved factor snapshots in a directory."""
    if not directory.exists():
        return []
    infos: list[FactorSnapshotInfo] = []
    for path in sorted(directory.glob("*.pkl")):
        try:
            snapshot = load_factor_snapshot(path)
        except (OSError, pickle.UnpicklingError, TypeError, ValueError):
            continue
        meta = snapshot.run.meta
        as_of = meta.get("as_of_date")
        train_end = meta.get("train_end")
        infos.append(
            FactorSnapshotInfo(
                key=path.stem,
                name=snapshot.name,
                created_at=snapshot.created_at,
                universe=snapshot.universe,
                weighting=snapshot.weighting,
                factor_set=snapshot.factor_set,
                start_date=snapshot.start_date,
                end_date=snapshot.end_date,
                garch_dist=snapshot.garch_dist,
                pca_demean=snapshot.pca_demean,
                train_fraction=snapshot.train_fraction,
                realized_window=snapshot.realized_window,
                n_assets=int(meta.get("n_assets", 0)),
                n_factors=int(meta.get("n_factors", 0)),
                n_pcs=int(meta.get("n_pcs", 0)),
                as_of_date=as_of.date() if hasattr(as_of, "date") else as_of,
                train_end=train_end.date() if hasattr(train_end, "date") else train_end,
            )
        )
    infos.sort(key=lambda item: item.created_at, reverse=True)
    return infos


__all__ = [
    "FactorAnalysisSnapshot",
    "FactorSnapshotInfo",
    "list_factor_snapshots",
    "load_factor_snapshot",
    "save_factor_snapshot",
]
