from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LAT_COL_CANDIDATES = ("latitude", "lat")
LON_COL_CANDIDATES = ("longitude", "lon", "lng")
ID_COL_CANDIDATES = ("id", "image_id", "idx")
COUNTRY_COL_CANDIDATES = ("country", "country_code", "iso_country")
REGION_COL_CANDIDATES = ("region", "admin_1", "state", "province")
SUB_REGION_COL_CANDIDATES = ("sub_region", "sub-region", "admin_2", "county", "city")


def pick(col_names: Iterable[str], cols: Iterable[str]) -> str | None:
    cols_set = set(cols)
    for c in col_names:
        if c in cols_set:
            return c
    return None


def stratified_indices_by_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    n_target: int,
    bin_deg: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat_bin = np.floor((lats + 90.0) / bin_deg).astype(np.int64)
    lon_bin = np.floor((lons + 180.0) / bin_deg).astype(np.int64)
    bin_id = lat_bin * 10_000 + lon_bin

    unique_bins, inverse, counts = np.unique(bin_id, return_inverse=True, return_counts=True)
    n_bins = len(unique_bins)
    per_bin = max(1, n_target // n_bins)

    chosen: list[np.ndarray] = []
    for b in range(n_bins):
        idx = np.flatnonzero(inverse == b)
        k = min(per_bin, len(idx))
        if k < len(idx):
            idx = rng.choice(idx, size=k, replace=False)
        chosen.append(idx)
    sel = np.concatenate(chosen)

    if len(sel) > n_target:
        sel = rng.choice(sel, size=n_target, replace=False)
    elif len(sel) < n_target:
        remaining = np.setdiff1d(np.arange(len(lats)), sel, assume_unique=False)
        extra_k = min(n_target - len(sel), len(remaining))
        if extra_k > 0:
            extra = rng.choice(remaining, size=extra_k, replace=False)
            sel = np.concatenate([sel, extra])

    rng.shuffle(sel)
    return sel


def train_val_split_stratified(
    strata: np.ndarray, val_fraction: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(strata)
    is_val = np.zeros(n, dtype=bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(idx) < 2:
            continue
        k = max(1, int(round(len(idx) * val_fraction)))
        k = min(k, len(idx) - 1)
        pick = rng.choice(idx, size=k, replace=False)
        is_val[pick] = True
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    return train_idx, val_idx


def load_metadata_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
