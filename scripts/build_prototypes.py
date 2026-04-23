from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prototypes import build_prototypes, save_prototypes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build location prototypes per geocell")
    parser.add_argument("--emb", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/prototypes.npz")
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

    proto = build_prototypes(
        embeddings=emb,
        lats=lats,
        lons=lons,
        cell_of_index=cell_of_index,
        train_mask=train_mask,
        max_k_per_cell=args.max_k,
        samples_per_prototype=args.samples_per_prototype,
        seed=args.seed,
    )
    print(
        f"[proto] built {len(proto['cell_id'])} prototypes over "
        f"{len(np.unique(proto['cell_id']))} cells"
    )
    save_prototypes(proto, args.out)
    print(f"[proto] saved to {args.out}")


if __name__ == "__main__":
    main()
