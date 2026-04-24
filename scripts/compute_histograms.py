from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.color_hist import DEFAULT_BINS, DEFAULT_GRID, compute_hsv_hist, hist_dim  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-image HSV histograms aligned with meta.parquet rows."
    )
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/image_histograms.npy")
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="Root for relative image paths. Defaults to data/osv5m_subset.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        nargs=3,
        default=list(DEFAULT_BINS),
        help="H S V bin counts (default 8 8 8).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=DEFAULT_GRID,
        help="Spatial grid size (grid x grid cells; default 3).",
    )
    args = parser.parse_args()

    meta = pd.read_parquet(args.meta)
    images_root = Path(args.images_root) if args.images_root else Path("data/osv5m_subset")
    if not images_root.exists():
        fallback = Path(args.meta).parent / "osv5m_subset"
        if fallback.exists():
            images_root = fallback
    print(f"[hist] images_root = {images_root}")

    bins = tuple(args.bins)
    d = hist_dim(bins, args.grid)
    hists = np.zeros((len(meta), d), dtype=np.float32)

    missing = 0
    for i, row in enumerate(tqdm(meta.itertuples(index=False), total=len(meta), desc="hist")):
        img_path = images_root / row.path
        try:
            img = Image.open(img_path)
            hists[i] = compute_hsv_hist(img, bins=bins, grid=args.grid)
            img.close()
        except Exception as e:
            missing += 1
            if missing <= 5:
                print(f"[hist] failed {img_path}: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, hists)
    print(
        f"[hist] saved shape={hists.shape} dtype={hists.dtype} "
        f"bins={bins} grid={args.grid} missing={missing}"
    )
    print(f"[hist] -> {out_path}")


if __name__ == "__main__":
    main()
