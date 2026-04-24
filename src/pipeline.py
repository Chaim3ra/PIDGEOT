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
from src.color_hist import DEFAULT_BINS, DEFAULT_GRID, compute_hsv_hist, intersection_similarity
from src.country_scripts import country_scripts, script_bonus
from src.embedder import StreetCLIPEmbedder, pick_device
from src.ocr import DEFAULT_LANGUAGES, ScriptDetector
from src.prototypes import load_prototypes


@dataclass
class PipelineArtifacts:
    embedder: StreetCLIPEmbedder
    classifier: torch.nn.Module
    ckpt: dict
    prototypes: dict[str, np.ndarray]
    device: torch.device
    script_detector: ScriptDetector | None


_state: dict[str, Any] = {"artifacts": None, "config_hash": None}
_lock = Lock()


def load_artifacts(
    classifier_path: str | Path,
    prototypes_path: str | Path,
    model_name: str = "geolocal/StreetCLIP",
    enable_ocr: bool = True,
    ocr_languages: tuple[str, ...] = DEFAULT_LANGUAGES,
) -> PipelineArtifacts:
    key = (str(classifier_path), str(prototypes_path), model_name, bool(enable_ocr), tuple(ocr_languages))
    with _lock:
        if _state["artifacts"] is not None and _state["config_hash"] == key:
            return _state["artifacts"]
        device = pick_device()
        embedder = StreetCLIPEmbedder(model_name=model_name, device=device)
        classifier, ckpt = load_classifier(classifier_path, map_location=device)
        classifier = classifier.to(device).eval()
        protos = load_prototypes(prototypes_path)
        script_detector = ScriptDetector.get(languages=ocr_languages) if enable_ocr else None
        arts = PipelineArtifacts(
            embedder=embedder,
            classifier=classifier,
            ckpt=ckpt,
            prototypes=protos,
            device=device,
            script_detector=script_detector,
        )
        _state["artifacts"] = arts
        _state["config_hash"] = key
        return arts


def _compute_script_bonus_vec(
    detected_scripts: set[str],
    candidate_countries: np.ndarray,
) -> np.ndarray:
    """Per-candidate re-rank bonus vector. See src.country_scripts.script_bonus for rules."""
    bonus = np.zeros(len(candidate_countries), dtype=np.float32)
    for i, cc in enumerate(candidate_countries):
        cc_str = str(cc) if cc is not None else ""
        if not cc_str:
            continue
        bonus[i] = script_bonus(detected_scripts, country_scripts(cc_str))
    return bonus


def _hist_params_from_protos(protos: dict[str, np.ndarray]) -> tuple[tuple[int, int, int], int]:
    if "hist_bins" in protos and "hist_grid" in protos:
        bins = tuple(int(b) for b in protos["hist_bins"].tolist())
        grid = int(protos["hist_grid"][0])
        return bins, grid  # type: ignore[return-value]
    return DEFAULT_BINS, DEFAULT_GRID


def predict(
    image: Image.Image,
    *,
    classifier_path: str | Path,
    prototypes_path: str | Path,
    top_cells: int = 5,
    top_k: int = 5,
    model_name: str = "geolocal/StreetCLIP",
    enable_ocr: bool = True,
    enable_color: bool = True,
    alpha: float = 1.0,
    # Defaults mirror the tuned `rerank` values in config.yaml so programmatic
    # callers (not just the Gradio UI) get the same behavior the tune script
    # selected on the val set. beta=0 = OCR neutral; gamma=0.2 = modest color
    # contribution. CLI tools / the UI override via config.yaml or explicit
    # kwargs.
    beta: float = 0.0,
    gamma: float = 0.2,
) -> dict:
    arts = load_artifacts(
        classifier_path, prototypes_path, model_name=model_name, enable_ocr=enable_ocr
    )
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
    proto_country = arts.prototypes.get("country", np.array([""] * len(proto_cell), dtype="<U8"))
    proto_hist = arts.prototypes.get("hist", None)

    sel_mask = np.isin(proto_cell, top_cell_ids)
    if not sel_mask.any():
        sel_mask = np.ones_like(proto_cell, dtype=bool)

    sel_emb = proto_emb[sel_mask]
    sel_cell = proto_cell[sel_mask]
    sel_lat = proto_lat[sel_mask]
    sel_lon = proto_lon[sel_mask]
    sel_country = proto_country[sel_mask]
    sel_hist = proto_hist[sel_mask] if proto_hist is not None else None

    sims = (sel_emb @ emb).astype(np.float32)

    detected: dict | None = None
    script_bonus_vec = np.zeros(len(sims), dtype=np.float32)
    if enable_ocr and arts.script_detector is not None:
        detected = arts.script_detector.detect(image)
        if detected["dominant"] is not None:
            script_bonus_vec = _compute_script_bonus_vec(set(detected["scripts"].keys()), sel_country)

    color_sim = np.zeros(len(sims), dtype=np.float32)
    color_active = bool(enable_color and sel_hist is not None and len(sel_hist) > 0)
    if color_active:
        bins, grid = _hist_params_from_protos(arts.prototypes)
        q_hist = compute_hsv_hist(image, bins=bins, grid=grid)
        color_sim = intersection_similarity(q_hist, sel_hist, bins=bins, grid=grid)

    final_score = (
        float(alpha) * sims
        + float(beta) * script_bonus_vec
        + float(gamma) * color_sim
    )

    order = np.argsort(-final_score)
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
                "script_bonus": (
                    float(script_bonus_vec[idx]) if detected and detected["dominant"] else None
                ),
                "color_sim": float(color_sim[idx]) if color_active else None,
                "final_score": float(final_score[idx]),
                "cell_id": int(sel_cell[idx]),
                "country": str(sel_country[idx]) if sel_country[idx] else "",
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
        "detected_scripts": detected["scripts"] if (detected and detected["dominant"]) else None,
        "ocr_enabled": bool(enable_ocr),
        "color_enabled": bool(color_active),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma) if color_active else 0.0,
    }
