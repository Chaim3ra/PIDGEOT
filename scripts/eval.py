from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import load_classifier  # noqa: E402
from src.embedder import pick_device  # noqa: E402
from src.metrics import haversine_km  # noqa: E402
from src.prototypes import load_prototypes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the held-out val split")
    parser.add_argument("--emb", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--protos", type=str, required=True)
    parser.add_argument("--top-cells", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    emb = np.load(args.emb)
    meta = pd.read_parquet(args.meta)
    assert len(emb) == len(meta)

    val_mask = meta["split"].to_numpy() == "val"
    val_idx = np.flatnonzero(val_mask)
    emb_val = emb[val_idx]
    lats_val = meta["lat"].to_numpy()[val_idx]
    lons_val = meta["lon"].to_numpy()[val_idx]
    cell_true = meta["geocell_id"].to_numpy()[val_idx].astype(np.int64)

    device = pick_device()
    classifier, ckpt = load_classifier(args.classifier, map_location=device)
    classifier = classifier.to(device).eval()
    centroids = ckpt["cell_centroids_latlon"]

    with torch.no_grad():
        logits = classifier(torch.from_numpy(emb_val).to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    preds_cell = probs.argmax(axis=1)
    top1 = float((preds_cell == cell_true).mean())
    top5 = float((np.argsort(-probs, axis=1)[:, :5] == cell_true[:, None]).any(axis=1).mean())

    pred_lat_cell = centroids[preds_cell, 0]
    pred_lon_cell = centroids[preds_cell, 1]
    km_err_cell = haversine_km(pred_lat_cell, pred_lon_cell, lats_val, lons_val)

    protos = load_prototypes(args.protos)
    proto_emb = protos["emb"]
    proto_cell = protos["cell_id"]
    proto_lat = protos["lat"]
    proto_lon = protos["lon"]

    top_cells = args.top_cells
    pred_lats_proto = np.zeros(len(emb_val), dtype=np.float64)
    pred_lons_proto = np.zeros(len(emb_val), dtype=np.float64)
    for i in range(len(emb_val)):
        top_cell_ids = np.argpartition(-probs[i], kth=top_cells - 1)[:top_cells]
        mask = np.isin(proto_cell, top_cell_ids)
        if not mask.any():
            pred_lats_proto[i] = pred_lat_cell[i]
            pred_lons_proto[i] = pred_lon_cell[i]
            continue
        sims = proto_emb[mask] @ emb_val[i]
        best = int(np.argmax(sims))
        pred_lats_proto[i] = proto_lat[mask][best]
        pred_lons_proto[i] = proto_lon[mask][best]
    km_err_proto = haversine_km(pred_lats_proto, pred_lons_proto, lats_val, lons_val)

    thresholds_km = [1, 25, 200, 750, 2500]
    frac_within = {f"frac_within_{t}km": float((km_err_proto < t).mean()) for t in thresholds_km}

    print(f"[eval] val n={len(emb_val)}")
    print(f"[eval] cell top-1 {top1:.3f} | top-5 {top5:.3f}")
    print(
        f"[eval] km error (cell-centroid pred): median {np.median(km_err_cell):.1f} "
        f"mean {np.mean(km_err_cell):.1f}"
    )
    print(
        f"[eval] km error (prototype refined): median {np.median(km_err_proto):.1f} "
        f"mean {np.mean(km_err_proto):.1f}"
    )
    for t in thresholds_km:
        print(f"[eval] frac within {t:>5} km: {frac_within[f'frac_within_{t}km']:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(lons_val, lats_val, c=np.log10(np.clip(km_err_proto, 1, None)), s=2, cmap="viridis")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("val errors (log10 km)")
    fig.colorbar(sc, ax=ax, label="log10 km")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_error_map.png", dpi=120)
    plt.close(fig)
    print(f"[eval] saved {out_dir / 'eval_error_map.png'}")

    metrics = {
        "val_n": int(len(emb_val)),
        "top_cells": int(args.top_cells),
        "cell_top1": top1,
        "cell_top5": top5,
        "km_err_cell_median": float(np.median(km_err_cell)),
        "km_err_cell_mean": float(np.mean(km_err_cell)),
        "km_err_proto_median": float(np.median(km_err_proto)),
        "km_err_proto_mean": float(np.mean(km_err_proto)),
        **frac_within,
    }
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] saved {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
