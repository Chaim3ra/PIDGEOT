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
from src.color_hist import DEFAULT_BINS, DEFAULT_GRID, intersection_similarity  # noqa: E402
from src.country_scripts import country_scripts, script_bonus  # noqa: E402
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
    parser.add_argument(
        "--ocr-cache",
        type=str,
        default="data/ocr_cache.parquet",
        help="Parquet file produced by scripts/tune_rerank.py. If absent, reranked metrics are skipped.",
    )
    parser.add_argument(
        "--image-histograms",
        type=str,
        default="data/image_histograms.npy",
        help="Per-image histogram cache. If absent, color-sim is skipped.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override rerank.beta; defaults to config.yaml rerank.beta if available.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Override rerank.gamma; defaults to config.yaml rerank.gamma if available.",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
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
    proto_country = protos.get("country", np.array([""] * len(proto_cell), dtype="<U8"))
    proto_hist = protos.get("hist", None)
    hist_bins = DEFAULT_BINS
    hist_grid = DEFAULT_GRID
    if proto_hist is not None and "hist_bins" in protos and "hist_grid" in protos:
        hist_bins = tuple(int(b) for b in protos["hist_bins"].tolist())
        hist_grid = int(protos["hist_grid"][0])

    # Per-image histogram cache for val queries (optional)
    image_hists = None
    image_hists_path = Path(args.image_histograms)
    if proto_hist is not None and image_hists_path.exists():
        loaded = np.load(image_hists_path)
        if len(loaded) == len(meta):
            image_hists = loaded
        else:
            print(
                f"[eval] WARNING: histogram rows {len(loaded)} != meta rows {len(meta)}; "
                "skipping color-sim"
            )

    # OCR cache (for reranked metrics)
    reranked_available = False
    ocr_cache_path = Path(args.ocr_cache)
    scripts_per_val: list[set[str]] = [set() for _ in range(len(emb_val))]
    dominant_per_val: list[str] = ["" for _ in range(len(emb_val))]
    if ocr_cache_path.exists():
        cache_df = pd.read_parquet(ocr_cache_path)
        cache_by_id = cache_df.set_index(cache_df["id"].astype(str))
        val_ids = meta.loc[val_mask, "id"].astype(str).tolist()
        covered = 0
        for i, id_str in enumerate(val_ids):
            if id_str in cache_by_id.index:
                row = cache_by_id.loc[id_str]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                dom = str(row["dominant_script"])
                if dom:
                    import json as _json
                    scripts_per_val[i] = set(_json.loads(row["scripts_json"]).keys())
                    dominant_per_val[i] = dom
                    covered += 1
        reranked_available = covered > 0
        print(f"[eval] OCR cache covers {covered}/{len(val_ids)} val rows")

    # Resolve beta and gamma from config (with CLI override)
    beta = args.beta
    gamma = args.gamma
    cfg_path = Path(args.config)
    if (beta is None or gamma is None) and cfg_path.exists():
        import yaml as _yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = _yaml.safe_load(f) or {}
        if beta is None:
            beta = float(raw.get("rerank", {}).get("beta", 0.5))
        if gamma is None:
            gamma = float(raw.get("rerank", {}).get("gamma", 0.3))
    if beta is None:
        beta = 0.5
    if gamma is None:
        gamma = 0.3
    color_available = proto_hist is not None and image_hists is not None

    top_cells = args.top_cells
    pred_lats_proto = np.zeros(len(emb_val), dtype=np.float64)
    pred_lons_proto = np.zeros(len(emb_val), dtype=np.float64)
    pred_lats_rerank = np.zeros(len(emb_val), dtype=np.float64)
    pred_lons_rerank = np.zeros(len(emb_val), dtype=np.float64)
    for i in range(len(emb_val)):
        top_cell_ids = np.argpartition(-probs[i], kth=top_cells - 1)[:top_cells]
        mask = np.isin(proto_cell, top_cell_ids)
        if not mask.any():
            pred_lats_proto[i] = pred_lat_cell[i]
            pred_lons_proto[i] = pred_lon_cell[i]
            pred_lats_rerank[i] = pred_lat_cell[i]
            pred_lons_rerank[i] = pred_lon_cell[i]
            continue
        sims = (proto_emb[mask] @ emb_val[i]).astype(np.float32)
        lats_p = proto_lat[mask]
        lons_p = proto_lon[mask]
        best = int(np.argmax(sims))
        pred_lats_proto[i] = lats_p[best]
        pred_lons_proto[i] = lons_p[best]

        bonus = np.zeros(len(sims), dtype=np.float32)
        if reranked_available and scripts_per_val[i]:
            countries = proto_country[mask]
            for k, cc in enumerate(countries):
                cc_str = str(cc)
                if not cc_str:
                    continue
                bonus[k] = script_bonus(scripts_per_val[i], country_scripts(cc_str))

        color = np.zeros(len(sims), dtype=np.float32)
        if color_available:
            q_hist = image_hists[val_idx[i]]
            color = intersection_similarity(q_hist, proto_hist[mask], bins=hist_bins, grid=hist_grid)

        score = sims + float(beta) * bonus + float(gamma) * color
        best_r = int(np.argmax(score))
        pred_lats_rerank[i] = lats_p[best_r]
        pred_lons_rerank[i] = lons_p[best_r]
    km_err_proto = haversine_km(pred_lats_proto, pred_lons_proto, lats_val, lons_val)
    km_err_rerank = haversine_km(pred_lats_rerank, pred_lons_rerank, lats_val, lons_val)

    thresholds_km = [1, 25, 200, 750, 2500]
    frac_within = {f"frac_within_{t}km": float((km_err_proto < t).mean()) for t in thresholds_km}
    frac_within_rerank = {
        f"frac_within_{t}km_reranked": float((km_err_rerank < t).mean()) for t in thresholds_km
    }

    # Per-subset metrics keyed on what OCR found. Useful for the writeup: the lift
    # from re-ranking is concentrated on the images where OCR actually fires, so
    # full-val medians understate the effect.
    dom_arr = np.array(dominant_per_val, dtype=object)
    subset_masks: dict[str, np.ndarray] = {
        "full_val": np.ones(len(emb_val), dtype=bool),
    }
    if reranked_available:
        subset_masks["ocr_fired"] = dom_arr != ""
        subset_masks["ocr_latin_only"] = dom_arr == "Latin"
        subset_masks["ocr_non_latin"] = (dom_arr != "") & (dom_arr != "Latin")

    subsets: dict[str, dict] = {}
    for name, m in subset_masks.items():
        n = int(m.sum())
        if n == 0:
            continue
        clip_err = km_err_proto[m]
        rerank_err = km_err_rerank[m]
        subsets[name] = {
            "n": n,
            "clip_only_median_km": float(np.median(clip_err)),
            "clip_only_mean_km": float(np.mean(clip_err)),
            "reranked_median_km": float(np.median(rerank_err)),
            "reranked_mean_km": float(np.mean(rerank_err)),
            "median_km_delta": float(np.median(clip_err) - np.median(rerank_err)),
            "median_km_rel_lift": (
                float((np.median(clip_err) - np.median(rerank_err)) / max(1e-9, np.median(clip_err)))
            ),
            **{
                f"clip_only_frac_within_{t}km": float((clip_err < t).mean()) for t in thresholds_km
            },
            **{
                f"reranked_frac_within_{t}km": float((rerank_err < t).mean()) for t in thresholds_km
            },
        }

    rerank_active = reranked_available or color_available
    print(f"[eval] val n={len(emb_val)}")
    print(f"[eval] cell top-1 {top1:.3f} | top-5 {top5:.3f}")
    print(
        f"[eval] km error (cell-centroid pred): median {np.median(km_err_cell):.1f} "
        f"mean {np.mean(km_err_cell):.1f}"
    )
    print(
        f"[eval] km error (prototype refined, CLIP only): median {np.median(km_err_proto):.1f} "
        f"mean {np.mean(km_err_proto):.1f}"
    )
    if rerank_active:
        beta_tag = f"beta={beta:.3f}" if reranked_available else "beta=- (no OCR cache)"
        gamma_tag = f"gamma={gamma:.3f}" if color_available else "gamma=- (no color hist)"
        print(
            f"[eval] km error (reranked, {beta_tag} {gamma_tag}): "
            f"median {np.median(km_err_rerank):.1f} mean {np.mean(km_err_rerank):.1f}"
        )
        print()
        print(f"[eval] per-subset breakdown (median km, CLIP-only -> reranked, rel. lift):")
        for name, s in subsets.items():
            print(
                f"  {name:>16s} n={s['n']:>5d}  "
                f"{s['clip_only_median_km']:>7.1f} -> {s['reranked_median_km']:>7.1f}  "
                f"({100*s['median_km_rel_lift']:+.1f}%)"
            )
    else:
        print(f"[eval] neither OCR cache nor histograms available — skipping rerank metrics")
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

    ocr_fire_rate = (
        float((dom_arr != "").sum() / max(1, len(dom_arr))) if reranked_available else None
    )
    metrics = {
        "val_n": int(len(emb_val)),
        "top_cells": int(args.top_cells),
        "cell_top1": top1,
        "cell_top5": top5,
        "km_err_cell_median": float(np.median(km_err_cell)),
        "km_err_cell_mean": float(np.mean(km_err_cell)),
        "km_err_proto_median": float(np.median(km_err_proto)),
        "km_err_proto_mean": float(np.mean(km_err_proto)),
        "km_err_proto_median_reranked": float(np.median(km_err_rerank)) if rerank_active else None,
        "km_err_proto_mean_reranked": float(np.mean(km_err_rerank)) if rerank_active else None,
        "rerank_beta": float(beta) if reranked_available else None,
        "rerank_gamma": float(gamma) if color_available else None,
        "ocr_fire_rate": ocr_fire_rate,
        "color_hist_available": bool(color_available),
        "subsets": subsets,
        **frac_within,
        **({k: v for k, v in frac_within_rerank.items()} if rerank_active else {}),
    }
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] saved {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
