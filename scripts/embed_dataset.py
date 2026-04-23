from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import train_val_split_stratified  # noqa: E402
from src.embedder import StreetCLIPEmbedder, pick_device  # noqa: E402
from src.geocells import load_cells  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed the subset with StreetCLIP")
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--cells", type=str, required=True)
    parser.add_argument("--out-emb", type=str, default="data/embeddings.npy")
    parser.add_argument("--out-meta", type=str, default="data/meta.parquet")
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto (64 GPU / 16 CPU)")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="Root for relative image paths. Defaults to directory containing metadata.csv.",
    )
    args = parser.parse_args()

    meta_path = Path(args.meta)
    df = pd.read_csv(meta_path)
    print(f"[embed] loaded {len(df)} rows")

    cells, cell_of_index = load_cells(args.cells)
    if len(cell_of_index) != len(df):
        raise RuntimeError(
            f"cells length {len(cell_of_index)} != metadata length {len(df)}; "
            "rebuild geocells against the current metadata.csv"
        )
    df = df.copy()
    df["geocell_id"] = cell_of_index

    train_idx, val_idx = train_val_split_stratified(
        cell_of_index, val_fraction=args.val_fraction, seed=args.seed
    )
    split = np.array(["train"] * len(df), dtype=object)
    split[val_idx] = "val"
    df["split"] = split
    print(f"[embed] split: {int((split == 'train').sum())} train / {int((split == 'val').sum())} val")

    device = pick_device()
    print(f"[embed] device: {device}")
    embedder = StreetCLIPEmbedder(device=device)
    if args.batch_size > 0:
        batch_size = args.batch_size
    else:
        batch_size = 64 if device.type == "cuda" else 16

    images_root = Path(args.images_root) if args.images_root else meta_path.parent
    embeddings = np.zeros((len(df), embedder.embed_dim), dtype=np.float32)

    total = len(df)
    with torch.inference_mode():
        for start in tqdm(range(0, total, batch_size), desc="embedding"):
            end = min(start + batch_size, total)
            rows = df.iloc[start:end]
            imgs = []
            for rel_path in rows["path"]:
                full = images_root / rel_path
                imgs.append(Image.open(full))
            vecs = embedder.embed_pil(imgs)
            for im in imgs:
                im.close()
            embeddings[start:end] = vecs

    np.save(args.out_emb, embeddings)
    print(f"[embed] saved embeddings to {args.out_emb} shape={embeddings.shape}")

    out_meta_path = Path(args.out_meta)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_meta_path, index=False)
    print(f"[embed] saved meta to {out_meta_path}")


if __name__ == "__main__":
    main()
