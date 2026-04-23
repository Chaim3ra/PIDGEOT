from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.geocells import build_semantic_geocells, save_cells  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic geocells from admin labels")
    parser.add_argument("--meta", type=str, required=True, help="metadata.csv from download_osv5m")
    parser.add_argument("--out", type=str, default="data/geocells/cells.pkl")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument(
        "--country-col",
        type=str,
        default="country",
        help="Column name for country; set to '' to skip.",
    )
    parser.add_argument("--region-col", type=str, default="region")
    parser.add_argument("--sub-region-col", type=str, default="sub_region")
    args = parser.parse_args()

    df = pd.read_csv(args.meta)
    print(f"[geocells] loaded {len(df)} rows from {args.meta}")
    print(f"[geocells] columns: {list(df.columns)}")

    cells, cell_of_index = build_semantic_geocells(
        df,
        min_samples=args.min_samples,
        country_col=args.country_col or None,
        region_col=args.region_col or None,
        sub_region_col=args.sub_region_col or None,
    )
    counts = np.array([c.member_count for c in cells])
    print(
        f"[geocells] built {len(cells)} cells | members min/med/max = "
        f"{int(counts.min())}/{int(np.median(counts))}/{int(counts.max())}"
    )

    save_cells(cells, cell_of_index, args.out)
    print(f"[geocells] saved to {args.out}")

    out_dir = Path(args.out).parent
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts, bins=50)
    ax.set_xlabel("members per cell")
    ax.set_ylabel("count")
    ax.set_title(f"Geocell size histogram (n_cells={len(cells)})")
    fig.tight_layout()
    fig.savefig(out_dir / "cell_size_hist.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    centroids = np.array([[c.centroid_lat, c.centroid_lon] for c in cells])
    ax.scatter(centroids[:, 1], centroids[:, 0], s=counts / counts.max() * 60 + 2, alpha=0.6)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(f"Geocell centroids (size ~ member count), n={len(cells)}")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cell_centroids.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
