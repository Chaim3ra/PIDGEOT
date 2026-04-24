from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

from src.color_hist import DEFAULT_BINS, DEFAULT_GRID


def build_prototypes(
    embeddings: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    cell_of_index: np.ndarray,
    train_mask: np.ndarray,
    max_k_per_cell: int = 10,
    samples_per_prototype: int = 50,
    seed: int = 42,
    cell_to_country: dict[int, str] | None = None,
    image_histograms: np.ndarray | None = None,
    hist_bins: Sequence[int] = DEFAULT_BINS,
    hist_grid: int = DEFAULT_GRID,
) -> dict[str, np.ndarray]:
    cell_ids_list: list[int] = []
    emb_list: list[np.ndarray] = []
    lat_list: list[float] = []
    lon_list: list[float] = []
    count_list: list[int] = []
    country_list: list[str] = []
    hist_list: list[np.ndarray] = []

    unique_cells = np.unique(cell_of_index)
    for cid in unique_cells:
        mask = (cell_of_index == cid) & train_mask
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        cell_emb = embeddings[idx]
        cell_lat = lats[idx]
        cell_lon = lons[idx]
        cell_hist = image_histograms[idx] if image_histograms is not None else None
        k = max(1, min(max_k_per_cell, len(idx) // samples_per_prototype))
        if k == 1 or len(idx) < 2:
            labels = np.zeros(len(idx), dtype=np.int64)
        else:
            km = KMeans(n_clusters=k, n_init=4, random_state=seed)
            labels = km.fit_predict(cell_emb)
        country = ""
        if cell_to_country is not None:
            country = str(cell_to_country.get(int(cid), "") or "")
        for c in range(k):
            m = labels == c
            if not m.any():
                continue
            mean_emb = cell_emb[m].mean(axis=0)
            mean_emb = mean_emb / max(1e-12, np.linalg.norm(mean_emb))
            cell_ids_list.append(int(cid))
            emb_list.append(mean_emb.astype(np.float32))
            lat_list.append(float(cell_lat[m].mean()))
            lon_list.append(float(cell_lon[m].mean()))
            count_list.append(int(m.sum()))
            country_list.append(country)
            if cell_hist is not None:
                hist_list.append(cell_hist[m].mean(axis=0).astype(np.float32))

    result: dict[str, np.ndarray] = {
        "cell_id": np.array(cell_ids_list, dtype=np.int64),
        "emb": np.stack(emb_list, axis=0) if emb_list else np.zeros((0, embeddings.shape[1]), np.float32),
        "lat": np.array(lat_list, dtype=np.float64),
        "lon": np.array(lon_list, dtype=np.float64),
        "count": np.array(count_list, dtype=np.int64),
        "country": np.array(country_list, dtype="<U8"),
    }
    if image_histograms is not None:
        result["hist"] = (
            np.stack(hist_list, axis=0)
            if hist_list
            else np.zeros((0, image_histograms.shape[1]), np.float32)
        )
        result["hist_bins"] = np.array(list(hist_bins), dtype=np.int32)
        result["hist_grid"] = np.array([int(hist_grid)], dtype=np.int32)
    return result


def save_prototypes(proto: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **proto)


def load_prototypes(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}
