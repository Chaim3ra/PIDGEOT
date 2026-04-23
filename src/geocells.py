from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.metrics import haversine_cross_km, spherical_centroid


UNKNOWN = "__UNK__"


def _normalize(value) -> str:
    if value is None:
        return UNKNOWN
    try:
        if isinstance(value, float) and np.isnan(value):
            return UNKNOWN
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return s if s and s.lower() != "nan" else UNKNOWN


@dataclass
class Cell:
    cell_id: int
    admin_path: tuple
    centroid_lat: float
    centroid_lon: float
    member_count: int
    row_indices: np.ndarray = field(repr=False)


def _country_to_continent(code: str) -> str:
    if not code or code == UNKNOWN:
        return UNKNOWN
    try:
        import pycountry_convert as pc

        alpha2 = code if len(code) == 2 else pc.country_name_to_country_alpha2(code)
        cont = pc.country_alpha2_to_continent_code(alpha2.upper())
        return cont
    except Exception:
        return UNKNOWN


def build_semantic_geocells(
    df: pd.DataFrame,
    min_samples: int = 30,
    lat_col: str = "lat",
    lon_col: str = "lon",
    country_col: str | None = "country",
    region_col: str | None = "region",
    sub_region_col: str | None = "sub_region",
) -> tuple[list[Cell], np.ndarray]:
    n = len(df)
    lats = df[lat_col].to_numpy(dtype=np.float64)
    lons = df[lon_col].to_numpy(dtype=np.float64)

    def col_or_unk(name: str | None) -> list[str]:
        if name and name in df.columns:
            return [_normalize(v) for v in df[name].tolist()]
        return [UNKNOWN] * n

    country = col_or_unk(country_col)
    region = col_or_unk(region_col)
    sub_region = col_or_unk(sub_region_col)

    group_to_indices: dict[tuple, list[int]] = {}
    for i in range(n):
        key = (country[i], region[i], sub_region[i])
        group_to_indices.setdefault(key, []).append(i)

    def merge_parents(idx_map: dict[tuple, list[int]], level: int) -> dict[tuple, list[int]]:
        parent_map: dict[tuple, list[int]] = {}
        for key, idxs in idx_map.items():
            if len(idxs) >= min_samples:
                # Big group: keep at this admin level. Use setdefault+extend so we don't
                # clobber indices from small groups that aggregated into this same key.
                parent_map.setdefault(key, []).extend(idxs)
            else:
                parent_key = key[:level] + (UNKNOWN,) * (len(key) - level)
                parent_map.setdefault(parent_key, []).extend(idxs)
        return parent_map

    merged = merge_parents(group_to_indices, level=2)
    merged = merge_parents(merged, level=1)

    tiny: list[tuple] = [k for k, v in merged.items() if len(v) < min_samples]
    big: list[tuple] = [k for k, v in merged.items() if len(v) >= min_samples]

    if big:
        big_centroids = np.array(
            [spherical_centroid(lats[np.array(merged[k])], lons[np.array(merged[k])]) for k in big]
        )
        big_continents = [_country_to_continent(k[0]) for k in big]
    else:
        big_centroids = np.zeros((0, 2))
        big_continents = []

    for k in tiny:
        idxs = merged.pop(k)
        if not big:
            merged.setdefault((UNKNOWN, UNKNOWN, UNKNOWN), []).extend(idxs)
            continue
        my_lat, my_lon = spherical_centroid(lats[np.array(idxs)], lons[np.array(idxs)])
        my_cont = _country_to_continent(k[0])
        cand_mask = np.array(
            [1 if my_cont == UNKNOWN or c == my_cont or c == UNKNOWN else 0 for c in big_continents]
        )
        if cand_mask.sum() == 0:
            cand_mask = np.ones(len(big), dtype=int)
        dists = haversine_cross_km(
            np.array([my_lat]), np.array([my_lon]), big_centroids[:, 0], big_centroids[:, 1]
        )[0]
        dists = np.where(cand_mask.astype(bool), dists, np.inf)
        target = big[int(np.argmin(dists))]
        merged[target].extend(idxs)

    cells: list[Cell] = []
    cell_of_index = np.full(n, -1, dtype=np.int64)
    for cid, (key, idxs) in enumerate(sorted(merged.items(), key=lambda kv: -len(kv[1]))):
        idxs_arr = np.array(sorted(idxs), dtype=np.int64)
        clat, clon = spherical_centroid(lats[idxs_arr], lons[idxs_arr])
        cells.append(
            Cell(
                cell_id=cid,
                admin_path=key,
                centroid_lat=clat,
                centroid_lon=clon,
                member_count=len(idxs_arr),
                row_indices=idxs_arr,
            )
        )
        cell_of_index[idxs_arr] = cid

    if (cell_of_index == -1).any():
        raise RuntimeError("Some rows were not assigned to any cell.")

    return cells, cell_of_index


def save_cells(cells: list[Cell], cell_of_index: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": [
            {
                "cell_id": c.cell_id,
                "admin_path": c.admin_path,
                "centroid_lat": c.centroid_lat,
                "centroid_lon": c.centroid_lon,
                "member_count": c.member_count,
            }
            for c in cells
        ],
        "cell_of_index": cell_of_index,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_cells(path: str | Path) -> tuple[list[dict], np.ndarray]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["cells"], payload["cell_of_index"]


def cell_centroids_array(cells: Iterable[dict]) -> np.ndarray:
    return np.array([[c["centroid_lat"], c["centroid_lon"]] for c in cells], dtype=np.float64)
