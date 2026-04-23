from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_matrix_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lats = np.radians(lats.astype(np.float64))
    lons = np.radians(lons.astype(np.float64))
    lat1 = lats[:, None]
    lat2 = lats[None, :]
    dlat = lat2 - lat1
    dlon = lons[None, :] - lons[:, None]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_cross_km(lats_a, lons_a, lats_b, lons_b) -> np.ndarray:
    lats_a = np.radians(np.asarray(lats_a, dtype=np.float64))
    lons_a = np.radians(np.asarray(lons_a, dtype=np.float64))
    lats_b = np.radians(np.asarray(lats_b, dtype=np.float64))
    lons_b = np.radians(np.asarray(lons_b, dtype=np.float64))
    dlat = lats_b[None, :] - lats_a[:, None]
    dlon = lons_b[None, :] - lons_a[:, None]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lats_a[:, None]) * np.cos(lats_b[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def top_k_accuracy(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    top_k = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    hits = (top_k == targets[:, None]).any(axis=1)
    return float(hits.mean())


def spherical_centroid(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    lats_r = np.radians(lats.astype(np.float64))
    lons_r = np.radians(lons.astype(np.float64))
    x = np.cos(lats_r) * np.cos(lons_r)
    y = np.cos(lats_r) * np.sin(lons_r)
    z = np.sin(lats_r)
    xm, ym, zm = x.mean(), y.mean(), z.mean()
    lon = np.degrees(np.arctan2(ym, xm))
    hyp = np.sqrt(xm * xm + ym * ym)
    lat = np.degrees(np.arctan2(zm, hyp))
    return float(lat), float(lon)
