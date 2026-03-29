"""Pickle-based persistence for regime detection snapshots, collections, and presets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import pickle
from typing import Optional

import pandas as pd

from .regime_detection import RegimeCollection, RegimeConfig, RegimeRun
from .style_storage import normalize_snapshot_name, snapshot_path


# ---------------------------------------------------------------------------
# Single regime snapshot (original)
# ---------------------------------------------------------------------------

@dataclass
class RegimeSnapshot:
    """Persisted regime detection snapshot."""

    name: str
    created_at: datetime
    series_name: str
    k_regimes: int
    preset: Optional[str]
    run: RegimeRun
    source_data: pd.Series

    @property
    def key(self) -> str:
        return normalize_snapshot_name(self.name)


@dataclass(frozen=True)
class RegimeSnapshotInfo:
    """Lightweight metadata for listing saved regime snapshots."""

    key: str
    name: str
    created_at: datetime
    series_name: str
    k_regimes: int
    preset: Optional[str]
    n_obs: int
    start_date: Optional[date]
    end_date: Optional[date]


def save_regime_snapshot(
    snapshot: RegimeSnapshot,
    directory: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a regime snapshot to a single pickle file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(directory, snapshot.name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Snapshot already exists: {snapshot.name}")
    with path.open("wb") as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_regime_snapshot(path: Path) -> RegimeSnapshot:
    """Load a regime snapshot from a pickle file."""
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, RegimeSnapshot):
        raise TypeError("Invalid snapshot payload.")
    return obj


def list_regime_snapshots(directory: Path) -> list[RegimeSnapshotInfo]:
    """List saved regime snapshots in a directory."""
    if not directory.exists():
        return []
    infos: list[RegimeSnapshotInfo] = []
    for path in sorted(directory.glob("*.pkl")):
        try:
            snapshot = load_regime_snapshot(path)
        except (OSError, pickle.UnpicklingError, TypeError, ValueError):
            continue
        meta = snapshot.run.meta
        infos.append(
            RegimeSnapshotInfo(
                key=path.stem,
                name=snapshot.name,
                created_at=snapshot.created_at,
                series_name=snapshot.series_name,
                k_regimes=snapshot.k_regimes,
                preset=snapshot.preset,
                n_obs=meta.get("n_obs", 0),
                start_date=_parse_date(meta.get("start_date")),
                end_date=_parse_date(meta.get("end_date")),
            )
        )
    infos.sort(key=lambda item: item.created_at, reverse=True)
    return infos


# ---------------------------------------------------------------------------
# Regime collection snapshot
# ---------------------------------------------------------------------------

@dataclass
class RegimeCollectionSnapshot:
    """Persisted regime collection (multiple regimes as one unit)."""

    name: str
    created_at: datetime
    collection: RegimeCollection

    @property
    def key(self) -> str:
        return normalize_snapshot_name(self.name)


@dataclass(frozen=True)
class RegimeCollectionInfo:
    """Lightweight metadata for listing saved collections."""

    key: str
    name: str
    created_at: datetime
    n_regimes: int
    regime_names: tuple[str, ...]


def save_regime_collection(
    snapshot: RegimeCollectionSnapshot,
    directory: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a regime collection snapshot."""
    subdir = directory / "collections"
    subdir.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(subdir, snapshot.name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Collection already exists: {snapshot.name}")
    with path.open("wb") as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_regime_collection(path: Path) -> RegimeCollectionSnapshot:
    """Load a regime collection snapshot."""
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, RegimeCollectionSnapshot):
        raise TypeError("Invalid collection snapshot payload.")
    return obj


def list_regime_collections(directory: Path) -> list[RegimeCollectionInfo]:
    """List saved regime collections."""
    subdir = directory / "collections"
    if not subdir.exists():
        return []
    infos: list[RegimeCollectionInfo] = []
    for path in sorted(subdir.glob("*.pkl")):
        try:
            snap = load_regime_collection(path)
        except (OSError, pickle.UnpicklingError, TypeError, ValueError):
            continue
        infos.append(
            RegimeCollectionInfo(
                key=path.stem,
                name=snap.name,
                created_at=snap.created_at,
                n_regimes=len(snap.collection),
                regime_names=tuple(snap.collection.names),
            )
        )
    infos.sort(key=lambda item: item.created_at, reverse=True)
    return infos


# ---------------------------------------------------------------------------
# Regime presets (user-saved configurations)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimePreset:
    """A user-saved regime configuration for quick reuse."""

    name: str
    description: str
    config: RegimeConfig
    created_at: datetime

    @property
    def key(self) -> str:
        return normalize_snapshot_name(self.name)


def save_regime_preset(
    preset: RegimePreset,
    directory: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a regime preset."""
    subdir = directory / "presets"
    subdir.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(subdir, preset.name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Preset already exists: {preset.name}")
    with path.open("wb") as handle:
        pickle.dump(preset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_regime_preset(path: Path) -> RegimePreset:
    """Load a regime preset."""
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, RegimePreset):
        raise TypeError("Invalid preset payload.")
    return obj


def list_regime_presets(directory: Path) -> list[RegimePreset]:
    """List saved regime presets."""
    subdir = directory / "presets"
    if not subdir.exists():
        return []
    presets: list[RegimePreset] = []
    for path in sorted(subdir.glob("*.pkl")):
        try:
            presets.append(load_regime_preset(path))
        except (OSError, pickle.UnpicklingError, TypeError, ValueError):
            continue
    presets.sort(key=lambda p: p.created_at, reverse=True)
    return presets


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_date(val: Optional[str]) -> Optional[date]:
    if val is None:
        return None
    try:
        return date.fromisoformat(val)
    except (TypeError, ValueError):
        return None


__all__ = [
    "RegimeCollectionInfo",
    "RegimeCollectionSnapshot",
    "RegimePreset",
    "RegimeSnapshot",
    "RegimeSnapshotInfo",
    "list_regime_collections",
    "list_regime_presets",
    "list_regime_snapshots",
    "load_regime_collection",
    "load_regime_preset",
    "load_regime_snapshot",
    "save_regime_collection",
    "save_regime_preset",
    "save_regime_snapshot",
]
