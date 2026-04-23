from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import haversine_cross_km


class GeocellHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_cells: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cells),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class HaversineSmoothingTargets:
    """Precomputed per-sample haversine distances used to build smoothed targets.

    `cell_dists_km[n, i]` = Hav(cell_centroid_i, sample_n_true_latlon)
    - Hav(true_cell_centroid_n, sample_n_true_latlon)
    i.e. already re-centered so the true cell's row is 0.
    """

    cell_dists_km: np.ndarray  # shape (N, num_cells)

    def soft_targets(self, indices: np.ndarray, tau_km: float) -> np.ndarray:
        d = self.cell_dists_km[indices]
        logits = -d / tau_km
        logits = logits - logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        w = w / w.sum(axis=1, keepdims=True)
        return w.astype(np.float32)


def build_smoothing_targets(
    sample_lats: np.ndarray,
    sample_lons: np.ndarray,
    true_cell_ids: np.ndarray,
    cell_centroids_latlon: np.ndarray,
) -> HaversineSmoothingTargets:
    cell_lats = cell_centroids_latlon[:, 0]
    cell_lons = cell_centroids_latlon[:, 1]
    d_sample_to_cells = haversine_cross_km(sample_lats, sample_lons, cell_lats, cell_lons)
    d_sample_to_true = d_sample_to_cells[np.arange(len(sample_lats)), true_cell_ids]
    centered = d_sample_to_cells - d_sample_to_true[:, None]
    return HaversineSmoothingTargets(cell_dists_km=centered.astype(np.float32))


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def save_classifier(
    model: GeocellHead,
    path: str,
    *,
    embed_dim: int,
    hidden_dim: int,
    num_cells: int,
    dropout: float,
    tau_km: float,
    cell_centroids_latlon: np.ndarray,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_cells": num_cells,
            "dropout": dropout,
            "tau_km": tau_km,
            "cell_centroids_latlon": cell_centroids_latlon,
        },
        path,
    )


def load_classifier(path: str, map_location: str | torch.device = "cpu") -> tuple[GeocellHead, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = GeocellHead(
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_cells=ckpt["num_cells"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
