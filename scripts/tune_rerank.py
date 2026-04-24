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
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import load_classifier  # noqa: E402
from src.color_hist import DEFAULT_BINS, DEFAULT_GRID, intersection_similarity  # noqa: E402
from src.country_scripts import country_scripts, script_bonus  # noqa: E402
from src.embedder import pick_device  # noqa: E402
from src.metrics import haversine_km  # noqa: E402
from src.ocr import DEFAULT_LANGUAGES, ScriptDetector  # noqa: E402
from src.prototypes import load_prototypes  # noqa: E402


def build_or_load_ocr_cache(
    cache_path: Path,
    meta: pd.DataFrame,
    images_root: Path,
    languages: tuple[str, ...],
    min_conf: float,
    min_chars: int,
    val_mask: np.ndarray,
) -> pd.DataFrame:
    val_ids = meta.loc[val_mask, "id"].astype(str).tolist()

    cached_ids: set[str] = set()
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached_ids = set(cached["id"].astype(str).tolist())
        print(f"[tune] cache hit: {len(cached_ids)} rows already OCR'd at {cache_path}")
    else:
        cached = pd.DataFrame(columns=["id", "dominant_script", "scripts_json"])

    missing = [i for i in val_ids if i not in cached_ids]
    if not missing:
        print("[tune] OCR cache fully covers val set.")
        return cached

    print(f"[tune] OCR'ing {len(missing)} new val images (cached: {len(cached_ids)})")
    detector = ScriptDetector.get(languages=languages)

    meta_by_id = meta.set_index(meta["id"].astype(str), drop=False)
    rows = []
    for id_str in tqdm(missing, desc="ocr"):
        rel = meta_by_id.loc[id_str, "path"]
        if isinstance(rel, pd.Series):
            rel = rel.iloc[0]
        img_path = images_root / rel
        try:
            img = Image.open(img_path)
            det = detector.detect(img, min_conf=min_conf, min_chars=min_chars)
            img.close()
        except Exception as e:
            print(f"[tune] OCR failed for {img_path}: {e}")
            det = {"scripts": {}, "text": [], "dominant": None}
        rows.append(
            {
                "id": id_str,
                "dominant_script": det["dominant"] or "",
                "scripts_json": json.dumps(det["scripts"]),
            }
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    combined = pd.concat([cached, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["id"], keep="last")
    combined.to_parquet(cache_path, index=False)
    print(f"[tune] wrote {len(combined)} OCR rows to {cache_path}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search OCR (beta) and color (gamma) rerank weights on val.")
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--emb", type=str, required=True)
    parser.add_argument("--cells", type=str, required=True)
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--protos", type=str, required=True)
    parser.add_argument("--cache", type=str, default="data/ocr_cache.parquet")
    parser.add_argument(
        "--image-histograms",
        type=str,
        default="data/image_histograms.npy",
        help="Per-image histograms aligned to meta rows. If missing, gamma sweep is skipped.",
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--top-cells", type=int, default=5)
    parser.add_argument("--images-root", type=str, default=None)
    parser.add_argument("--min-conf", type=float, default=0.4)
    parser.add_argument("--min-chars", type=int, default=3)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="If set, write best beta AND best gamma back to config.yaml.",
    )
    args = parser.parse_args()

    meta = pd.read_parquet(args.meta)
    emb = np.load(args.emb)
    if len(emb) != len(meta):
        raise RuntimeError(f"emb rows {len(emb)} != meta rows {len(meta)}")

    val_mask = (meta["split"].to_numpy() == "val")
    val_idx = np.flatnonzero(val_mask)
    lats_val = meta["lat"].to_numpy()[val_idx]
    lons_val = meta["lon"].to_numpy()[val_idx]

    images_root = (
        Path(args.images_root) if args.images_root else Path(args.meta).parent / "osv5m_subset"
    )
    if not images_root.exists():
        images_root = Path("data/osv5m_subset")

    cache_path = Path(args.cache)
    cache_df = build_or_load_ocr_cache(
        cache_path,
        meta,
        images_root,
        tuple(args.languages),
        args.min_conf,
        args.min_chars,
        val_mask,
    )
    cache_by_id = cache_df.set_index(cache_df["id"].astype(str))

    val_ids = meta.loc[val_mask, "id"].astype(str).tolist()
    dominant = []
    scripts_per_val: list[set[str]] = []
    for id_str in val_ids:
        if id_str in cache_by_id.index:
            row = cache_by_id.loc[id_str]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            dom = str(row["dominant_script"]) or ""
            scripts = json.loads(row["scripts_json"]) if row["scripts_json"] else {}
        else:
            dom, scripts = "", {}
        dominant.append(dom)
        scripts_per_val.append(set(scripts.keys()) if dom else set())
    ocr_fire_rate = float(sum(1 for d in dominant if d) / max(1, len(dominant)))
    print(f"[tune] OCR fire-rate on val: {ocr_fire_rate:.3f}")

    # Load per-image histograms for val, if available
    image_hists = None
    image_hists_path = Path(args.image_histograms)
    if image_hists_path.exists():
        image_hists = np.load(image_hists_path)
        if len(image_hists) != len(meta):
            print(
                f"[tune] WARNING: histogram rows {len(image_hists)} != meta rows {len(meta)}; "
                "skipping gamma sweep"
            )
            image_hists = None
        else:
            print(f"[tune] loaded per-image histograms shape={image_hists.shape}")
    else:
        print(f"[tune] no histogram cache at {image_hists_path}; skipping gamma sweep")

    device = pick_device()
    classifier, ckpt = load_classifier(args.classifier, map_location=device)
    classifier = classifier.to(device).eval()

    with torch.no_grad():
        logits = classifier(torch.from_numpy(emb[val_idx]).to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    protos = load_prototypes(args.protos)
    proto_cell = protos["cell_id"]
    proto_emb = protos["emb"]
    proto_lat = protos["lat"]
    proto_lon = protos["lon"]
    proto_country = protos.get("country", np.array([""] * len(proto_cell), dtype="<U8"))
    proto_hist = protos.get("hist", None)
    if proto_hist is None or len(proto_hist) == 0:
        print("[tune] prototypes.npz missing 'hist'; gamma sweep will have no effect")
        proto_hist = None

    hist_bins = DEFAULT_BINS
    hist_grid = DEFAULT_GRID
    if proto_hist is not None and "hist_bins" in protos and "hist_grid" in protos:
        hist_bins = tuple(int(b) for b in protos["hist_bins"].tolist())
        hist_grid = int(protos["hist_grid"][0])

    all_sims: list[np.ndarray] = [None] * len(val_idx)
    all_bonus: list[np.ndarray] = [None] * len(val_idx)
    all_color: list[np.ndarray] = [None] * len(val_idx)
    all_lat: list[np.ndarray] = [None] * len(val_idx)
    all_lon: list[np.ndarray] = [None] * len(val_idx)

    for i in tqdm(range(len(val_idx)), desc="score"):
        top_cell_ids = np.argpartition(-probs[i], kth=args.top_cells - 1)[: args.top_cells]
        mask = np.isin(proto_cell, top_cell_ids)
        if not mask.any():
            mask = np.ones_like(proto_cell, dtype=bool)
        sims = (proto_emb[mask] @ emb[val_idx[i]]).astype(np.float32)
        lats_p = proto_lat[mask]
        lons_p = proto_lon[mask]
        countries = proto_country[mask]

        detected = scripts_per_val[i]
        if detected:
            bonus = np.zeros(len(sims), dtype=np.float32)
            for k, cc in enumerate(countries):
                cc_str = str(cc)
                if not cc_str:
                    continue
                bonus[k] = script_bonus(detected, country_scripts(cc_str))
        else:
            bonus = np.zeros(len(sims), dtype=np.float32)

        if proto_hist is not None and image_hists is not None:
            q_hist = image_hists[val_idx[i]]
            color = intersection_similarity(q_hist, proto_hist[mask], bins=hist_bins, grid=hist_grid)
        else:
            color = np.zeros(len(sims), dtype=np.float32)

        all_sims[i] = sims
        all_bonus[i] = bonus
        all_color[i] = color
        all_lat[i] = lats_p
        all_lon[i] = lons_p

    thresholds_km = [1, 25, 200, 750, 2500]

    def evaluate(beta: float, gamma: float) -> dict[str, float]:
        pred_lat = np.zeros(len(val_idx), dtype=np.float64)
        pred_lon = np.zeros(len(val_idx), dtype=np.float64)
        for i in range(len(val_idx)):
            score = all_sims[i] + float(beta) * all_bonus[i] + float(gamma) * all_color[i]
            best = int(np.argmax(score))
            pred_lat[i] = all_lat[i][best]
            pred_lon[i] = all_lon[i][best]
        km_err = haversine_km(pred_lat, pred_lon, lats_val, lons_val)
        row = {"beta": float(beta), "gamma": float(gamma)}
        row["median_km"] = float(np.median(km_err))
        row["mean_km"] = float(np.mean(km_err))
        for t in thresholds_km:
            row[f"frac_within_{t}km"] = float((km_err < t).mean())
        return row

    # 1D sweep: beta with gamma=0
    beta_rows = []
    print("\n[tune] 1D sweep: beta (gamma=0)")
    for b in args.betas:
        row = evaluate(b, 0.0)
        beta_rows.append(row)
        print(f"[tune] beta={b:.3f} gamma=0.000 | median_km={row['median_km']:.1f} mean_km={row['mean_km']:.1f}")
    best_beta_row = min(beta_rows, key=lambda r: r["median_km"])
    print(f"[tune] best beta (gamma=0) = {best_beta_row['beta']:.3f} (median_km={best_beta_row['median_km']:.1f})")

    # 1D sweep: gamma with beta=0
    gamma_rows = []
    if proto_hist is not None and image_hists is not None:
        print("\n[tune] 1D sweep: gamma (beta=0)")
        for g in args.gammas:
            row = evaluate(0.0, g)
            gamma_rows.append(row)
            print(f"[tune] beta=0.000 gamma={g:.3f} | median_km={row['median_km']:.1f} mean_km={row['mean_km']:.1f}")
        best_gamma_row = min(gamma_rows, key=lambda r: r["median_km"])
        print(
            f"[tune] best gamma (beta=0) = {best_gamma_row['gamma']:.3f} "
            f"(median_km={best_gamma_row['median_km']:.1f})"
        )
    else:
        best_gamma_row = {"beta": 0.0, "gamma": 0.0, "median_km": best_beta_row["median_km"]}

    # 2D sweep: all (beta, gamma) combinations
    joint_rows = []
    grid_2d = None
    if proto_hist is not None and image_hists is not None:
        print("\n[tune] 2D sweep: beta x gamma")
        grid_2d = np.zeros((len(args.betas), len(args.gammas)), dtype=np.float32)
        for bi, b in enumerate(args.betas):
            for gi, g in enumerate(args.gammas):
                row = evaluate(b, g)
                joint_rows.append(row)
                grid_2d[bi, gi] = row["median_km"]
        best_joint = min(joint_rows, key=lambda r: r["median_km"])
        print(
            f"[tune] best joint (beta={best_joint['beta']:.3f}, gamma={best_joint['gamma']:.3f}) "
            f"median_km={best_joint['median_km']:.1f}"
        )
    else:
        best_joint = best_beta_row

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: beta sweep
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [r["beta"] for r in beta_rows]
    ys = [r["median_km"] for r in beta_rows]
    ax.plot(xs, ys, marker="o")
    ax.axvline(
        best_beta_row["beta"], color="red", linestyle="--", alpha=0.5,
        label=f"best beta={best_beta_row['beta']:.2f}",
    )
    ax.set_xlabel("beta (OCR weight)")
    ax.set_ylabel("median haversine km error (val)")
    ax.set_title(f"OCR re-rank grid search (n_val={len(val_idx)}, ocr_fire={ocr_fire_rate:.2f})")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "rerank_gridsearch.png", dpi=120)
    plt.close(fig)
    print(f"[tune] saved {results_dir / 'rerank_gridsearch.png'}")

    # Plot 2: gamma sweep
    if gamma_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = [r["gamma"] for r in gamma_rows]
        ys = [r["median_km"] for r in gamma_rows]
        ax.plot(xs, ys, marker="o", color="tab:green")
        ax.axvline(
            best_gamma_row["gamma"], color="red", linestyle="--", alpha=0.5,
            label=f"best gamma={best_gamma_row['gamma']:.2f}",
        )
        ax.set_xlabel("gamma (color-hist weight)")
        ax.set_ylabel("median haversine km error (val)")
        ax.set_title(f"Color-hist re-rank grid search (n_val={len(val_idx)})")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "rerank_gridsearch_gamma.png", dpi=120)
        plt.close(fig)
        print(f"[tune] saved {results_dir / 'rerank_gridsearch_gamma.png'}")

    # Plot 3: 2D heatmap
    if grid_2d is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid_2d, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(args.gammas)))
        ax.set_xticklabels([f"{g:.2f}" for g in args.gammas], rotation=45)
        ax.set_yticks(range(len(args.betas)))
        ax.set_yticklabels([f"{b:.2f}" for b in args.betas])
        ax.set_xlabel("gamma (color)")
        ax.set_ylabel("beta (OCR)")
        ax.set_title("median haversine km error (val) over beta x gamma")
        # Mark best
        bi = args.betas.index(best_joint["beta"])
        gi = args.gammas.index(best_joint["gamma"])
        ax.plot(gi, bi, marker="*", color="red", markersize=18, markeredgecolor="white", markeredgewidth=1)
        fig.colorbar(im, ax=ax, label="median km")
        fig.tight_layout()
        fig.savefig(results_dir / "rerank_gridsearch_2d.png", dpi=120)
        plt.close(fig)
        print(f"[tune] saved {results_dir / 'rerank_gridsearch_2d.png'}")

    metrics = {
        "val_n": int(len(val_idx)),
        "top_cells": int(args.top_cells),
        "ocr_fire_rate": ocr_fire_rate,
        "languages": list(args.languages),
        "min_conf": float(args.min_conf),
        "min_chars": int(args.min_chars),
        "color_hist_dim": int(proto_hist.shape[1]) if proto_hist is not None else 0,
        "per_beta": {f"{r['beta']:.3f}": r for r in beta_rows},
        "per_gamma": {f"{r['gamma']:.3f}": r for r in gamma_rows},
        "best_beta": float(best_beta_row["beta"]),
        "best_gamma": float(best_gamma_row["gamma"]),
        "best_joint_beta": float(best_joint["beta"]),
        "best_joint_gamma": float(best_joint.get("gamma", 0.0)),
        "best_joint_median_km": float(best_joint["median_km"]),
    }
    with open(results_dir / "rerank_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[tune] saved {results_dir / 'rerank_metrics.json'}")

    if args.update_config:
        cfg_path = Path(args.config)
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw.setdefault("rerank", {})["beta"] = float(best_joint["beta"])
        raw["rerank"]["gamma"] = float(best_joint.get("gamma", 0.0))
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        print(
            f"[tune] updated {cfg_path} rerank.beta={best_joint['beta']:.3f} "
            f"rerank.gamma={best_joint.get('gamma', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
