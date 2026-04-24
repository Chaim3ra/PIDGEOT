from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.color_hist import DEFAULT_BINS, DEFAULT_GRID  # noqa: E402
from src.geocells import load_cells  # noqa: E402
from src.prototypes import build_prototypes, save_prototypes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build location prototypes per geocell")
    parser.add_argument("--emb", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/prototypes.npz")
    parser.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Optional cells.pkl. When given, each prototype is tagged with its country "
             "(from admin_path[0]) so the OCR re-ranker can look up acceptable scripts.",
    )
    parser.add_argument(
        "--image-histograms",
        type=str,
        default=None,
        help="Optional path to per-image HSV histograms npy (aligned to meta rows). "
             "When given, each prototype stores a mean-of-members histogram for color re-ranking.",
    )
    parser.add_argument("--hist-bins", type=int, nargs=3, default=list(DEFAULT_BINS))
    parser.add_argument("--hist-grid", type=int, default=DEFAULT_GRID)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--samples-per-prototype", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    emb = np.load(args.emb)
    meta = pd.read_parquet(args.meta)
    if len(emb) != len(meta):
        raise RuntimeError(f"emb rows {len(emb)} != meta rows {len(meta)}")

    cell_of_index = meta["geocell_id"].to_numpy(dtype=np.int64)
    lats = meta["lat"].to_numpy(dtype=np.float64)
    lons = meta["lon"].to_numpy(dtype=np.float64)
    train_mask = (meta["split"].to_numpy() == "train")

    cell_to_country: dict[int, str] | None = None
    if args.cells:
        cells, _ = load_cells(args.cells)
        cell_to_country = {}
        for c in cells:
            admin = c.get("admin_path") or ()
            cc = admin[0] if len(admin) > 0 else ""
            cell_to_country[int(c["cell_id"])] = str(cc) if cc and cc != "__UNK__" else ""
        nonempty = sum(1 for v in cell_to_country.values() if v)
        print(f"[proto] loaded country map for {nonempty}/{len(cell_to_country)} cells")

    image_hists = None
    if args.image_histograms:
        image_hists = np.load(args.image_histograms)
        if len(image_hists) != len(meta):
            raise RuntimeError(
                f"image_histograms rows {len(image_hists)} != meta rows {len(meta)}"
            )
        print(f"[proto] loaded image histograms shape={image_hists.shape}")

    proto = build_prototypes(
        embeddings=emb,
        lats=lats,
        lons=lons,
        cell_of_index=cell_of_index,
        train_mask=train_mask,
        max_k_per_cell=args.max_k,
        samples_per_prototype=args.samples_per_prototype,
        seed=args.seed,
        cell_to_country=cell_to_country,
        image_histograms=image_hists,
        hist_bins=tuple(args.hist_bins),
        hist_grid=args.hist_grid,
    )
    print(
        f"[proto] built {len(proto['cell_id'])} prototypes over "
        f"{len(np.unique(proto['cell_id']))} cells"
    )
    if cell_to_country is not None:
        labeled = int(np.sum(proto["country"] != ""))
        print(f"[proto] {labeled}/{len(proto['cell_id'])} prototypes have a country label")
    if image_hists is not None:
        print(
            f"[proto] attached color histograms: shape={proto['hist'].shape} "
            f"bins={proto['hist_bins'].tolist()} grid={int(proto['hist_grid'][0])}"
        )
    save_prototypes(proto, args.out)
    print(f"[proto] saved to {args.out}")


if __name__ == "__main__":
    main()
