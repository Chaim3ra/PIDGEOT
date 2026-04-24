"""HSV color histograms with spatial grid, for prototype re-ranking.

Each image is partitioned into a `grid x grid` set of rectangular cells, and
a 3D HSV histogram (bins[0] hue x bins[1] saturation x bins[2] value) is
computed per cell and L1-normalized. The cell histograms are concatenated
into one flat vector of length `grid*grid*bins[0]*bins[1]*bins[2]`.

Default of 3x3 spatial cells (9 regions) captures palette layout — e.g.,
"blue on top, green in middle, grey at bottom" — which a single global
histogram throws away. Bin count 8*8*8 = 512 per cell is the standard
Project-2 choice; total dim = 4608.

Similarity between two images uses per-cell histogram intersection,
averaged across the 9 cells, yielding a score in [0, 1].
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from PIL import Image


DEFAULT_BINS: tuple[int, int, int] = (8, 8, 8)
DEFAULT_GRID: int = 3


def hist_dim(bins: Sequence[int] = DEFAULT_BINS, grid: int = DEFAULT_GRID) -> int:
    return int(grid * grid * bins[0] * bins[1] * bins[2])


def compute_hsv_hist(
    image: Image.Image | np.ndarray,
    bins: Sequence[int] = DEFAULT_BINS,
    grid: int = DEFAULT_GRID,
) -> np.ndarray:
    """Compute the spatial-grid HSV histogram of an image.

    Returns a flat float32 array of length `hist_dim(bins, grid)`, with each
    per-cell block L1-normalized so cells with identical distributions match
    regardless of cell pixel count.
    """
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[-1] == 4:
            rgb = arr[..., :3]
        else:
            rgb = arr
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    per_cell = int(bins[0] * bins[1] * bins[2])
    result = np.zeros(grid * grid * per_cell, dtype=np.float32)

    cell_h = h / grid
    cell_w = w / grid
    for i in range(grid):
        y0 = int(round(i * cell_h))
        y1 = int(round((i + 1) * cell_h)) if i < grid - 1 else h
        for j in range(grid):
            x0 = int(round(j * cell_w))
            x1 = int(round((j + 1) * cell_w)) if j < grid - 1 else w
            cell = hsv[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            # OpenCV HSV ranges: H [0, 180), S [0, 256), V [0, 256).
            hist = cv2.calcHist(
                [cell],
                channels=[0, 1, 2],
                mask=None,
                histSize=list(bins),
                ranges=[0, 180, 0, 256, 0, 256],
            )
            total = float(hist.sum())
            if total > 0:
                hist = hist / total
            cell_idx = i * grid + j
            offset = cell_idx * per_cell
            result[offset : offset + per_cell] = hist.reshape(-1).astype(np.float32)
    return result


def intersection_similarity(
    q: np.ndarray,
    protos: np.ndarray,
    bins: Sequence[int] = DEFAULT_BINS,
    grid: int = DEFAULT_GRID,
) -> np.ndarray:
    """Mean per-cell histogram-intersection similarity between one query and N prototypes.

    q      : shape (D,) or (1, D)   — flat histogram for the query image
    protos : shape (N, D)           — stacked flat histograms for candidate prototypes
    returns: shape (N,)             — similarity in [0, 1]
    """
    if protos.size == 0:
        return np.zeros(0, dtype=np.float32)
    q2 = np.asarray(q, dtype=np.float32)
    if q2.ndim == 1:
        q2 = q2[None, :]
    n_cells = int(grid * grid)
    per_cell = int(bins[0] * bins[1] * bins[2])
    if q2.shape[-1] != n_cells * per_cell or protos.shape[-1] != n_cells * per_cell:
        raise ValueError(
            f"Histogram dim mismatch: q={q2.shape[-1]}, protos={protos.shape[-1]}, "
            f"expected {n_cells * per_cell} for bins={tuple(bins)}, grid={grid}"
        )
    q_cells = q2.reshape(q2.shape[0], n_cells, per_cell)
    p_cells = protos.reshape(protos.shape[0], n_cells, per_cell)
    # Broadcast: (1, C, D) vs (N, C, D) -> min -> (N, C, D)
    inter = np.minimum(q_cells, p_cells).sum(axis=-1)  # (N, C)
    return inter.mean(axis=-1).astype(np.float32)
