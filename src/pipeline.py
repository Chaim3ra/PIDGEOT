from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.classifier import load_classifier
from src.embedder import StreetCLIPEmbedder, pick_device
from src.prototypes import load_prototypes


@dataclass
class PipelineArtifacts:
    embedder: StreetCLIPEmbedder
    classifier: torch.nn.Module
    ckpt: dict
    prototypes: dict[str, np.ndarray]
    device: torch.device


_state: dict[str, Any] = {"artifacts": None, "config_hash": None}
_lock = Lock()


def load_artifacts(
    classifier_path: str | Path,
    prototypes_path: str | Path,
    model_name: str = "geolocal/StreetCLIP",
) -> PipelineArtifacts:
    key = (str(classifier_path), str(prototypes_path), model_name)
    with _lock:
        if _state["artifacts"] is not None and _state["config_hash"] == key:
            return _state["artifacts"]
        device = pick_device()
        embedder = StreetCLIPEmbedder(model_name=model_name, device=device)
        classifier, ckpt = load_classifier(classifier_path, map_location=device)
        classifier = classifier.to(device).eval()
        protos = load_prototypes(prototypes_path)
        arts = PipelineArtifacts(
            embedder=embedder,
            classifier=classifier,
            ckpt=ckpt,
            prototypes=protos,
            device=device,
        )
        _state["artifacts"] = arts
        _state["config_hash"] = key
        return arts


def predict(
    image: Image.Image,
    *,
    classifier_path: str | Path,
    prototypes_path: str | Path,
    top_cells: int = 5,
    top_k: int = 5,
    model_name: str = "geolocal/StreetCLIP",
) -> dict:
    arts = load_artifacts(classifier_path, prototypes_path, model_name=model_name)
    emb = arts.embedder.embed_one(image)
    emb_t = torch.from_numpy(emb[None, :]).to(arts.device)

    with torch.no_grad():
        logits = arts.classifier(emb_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    num_cells = probs.shape[0]
    top_cells_eff = min(top_cells, num_cells)
    top_cell_ids = np.argpartition(-probs, kth=top_cells_eff - 1)[:top_cells_eff]
    top_cell_ids = top_cell_ids[np.argsort(-probs[top_cell_ids])]

    proto_cell = arts.prototypes["cell_id"]
    proto_emb = arts.prototypes["emb"]
    proto_lat = arts.prototypes["lat"]
    proto_lon = arts.prototypes["lon"]

    sel_mask = np.isin(proto_cell, top_cell_ids)
    if not sel_mask.any():
        sel_mask = np.ones_like(proto_cell, dtype=bool)

    sel_emb = proto_emb[sel_mask]
    sel_cell = proto_cell[sel_mask]
    sel_lat = proto_lat[sel_mask]
    sel_lon = proto_lon[sel_mask]

    sims = sel_emb @ emb
    order = np.argsort(-sims)
    top_k_eff = min(top_k, len(order))
    top = order[:top_k_eff]

    candidates = []
    for rank, idx in enumerate(top):
        candidates.append(
            {
                "rank": rank + 1,
                "lat": float(sel_lat[idx]),
                "lon": float(sel_lon[idx]),
                "confidence": float((sims[idx] + 1.0) / 2.0),
                "similarity": float(sims[idx]),
                "cell_id": int(sel_cell[idx]),
            }
        )

    centroids = arts.ckpt["cell_centroids_latlon"]
    cell_probs = [
        {
            "cell_id": int(cid),
            "prob": float(probs[cid]),
            "centroid_lat": float(centroids[cid, 0]),
            "centroid_lon": float(centroids[cid, 1]),
        }
        for cid in top_cell_ids.tolist()
    ]

    best = candidates[0]
    return {
        "predicted": (best["lat"], best["lon"]),
        "confidence": best["confidence"],
        "top_k": candidates,
        "cell_probs": cell_probs,
    }
