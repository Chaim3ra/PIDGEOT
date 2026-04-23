from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (  # noqa: E402
    COUNTRY_COL_CANDIDATES,
    ID_COL_CANDIDATES,
    LAT_COL_CANDIDATES,
    LON_COL_CANDIDATES,
    REGION_COL_CANDIDATES,
    SUB_REGION_COL_CANDIDATES,
    pick,
    stratified_indices_by_grid,
)


REPO_ID = "osv5m/osv5m"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a stratified subset of osv5m directly from the HF Hub (no dataset loader)."
    )
    parser.add_argument("--n", type=int, default=50_000, help="Target number of images")
    parser.add_argument("--out", type=str, default="data/osv5m_subset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "test"),
        help="osv5m split to pull from. 'test' has ~210k images across 5 shards (~450 MB-2.2 GB each); "
             "'train' has ~5M across 98 shards (~2.5 GB each). Default: test (ample for 50k and way smaller).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=2,
        help="Max number of zip shards to download (starting at 00). Each test shard ~42k images; "
             "each train shard ~51k. Set high enough to cover --n after stratified sampling.",
    )
    parser.add_argument("--bin-deg", type=float, default=10.0)
    parser.add_argument(
        "--keep-extra",
        action="store_true",
        help="Keep extracted images that weren't selected (useful to avoid re-extraction on re-runs).",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download  # lazy import

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[osv5m] downloading metadata: {args.split}.csv ...")
    meta_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"{args.split}.csv",
    )
    print(f"[osv5m] metadata at {meta_path}")
    df = pd.read_csv(meta_path, low_memory=False)
    cols = list(df.columns)
    print(f"[osv5m] metadata columns: {cols}")

    lat_col = pick(LAT_COL_CANDIDATES, cols)
    lon_col = pick(LON_COL_CANDIDATES, cols)
    id_col = pick(ID_COL_CANDIDATES, cols)
    country_col = pick(COUNTRY_COL_CANDIDATES, cols)
    region_col = pick(REGION_COL_CANDIDATES, cols)
    sub_region_col = pick(SUB_REGION_COL_CANDIDATES, cols)
    if lat_col is None or lon_col is None or id_col is None:
        raise RuntimeError(f"Could not find id/lat/lon in osv5m csv; saw {cols}")
    print(
        f"[osv5m] using id='{id_col}', lat='{lat_col}', lon='{lon_col}', "
        f"country='{country_col}', region='{region_col}', sub_region='{sub_region_col}'"
    )

    df["_id_str"] = df[id_col].astype(str).str.strip()

    downloaded_zips: list[Path] = []
    available_ids: set[str] = set()
    for shard_num in range(args.max_shards):
        shard_rel = f"images/{args.split}/{shard_num:02d}.zip"
        print(f"[osv5m] downloading shard {shard_rel} ...")
        try:
            zpath = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=shard_rel,
            )
        except Exception as e:
            print(f"[osv5m] shard {shard_rel} unavailable, stopping ({e})")
            break
        downloaded_zips.append(Path(zpath))
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".jpg") or n.lower().endswith(".jpeg")]
            print(f"[osv5m]   shard has {len(names)} images")
            for n in names:
                available_ids.add(Path(n).stem)

    if not available_ids:
        raise RuntimeError("No images available after downloading shards.")

    print(f"[osv5m] available ids from downloaded shards: {len(available_ids)}")
    avail_mask = df["_id_str"].isin(available_ids)
    df_avail = df[avail_mask].reset_index(drop=True)
    print(f"[osv5m] metadata rows joining to available images: {len(df_avail)}")

    lats = df_avail[lat_col].to_numpy(dtype=np.float64)
    lons = df_avail[lon_col].to_numpy(dtype=np.float64)
    valid = np.isfinite(lats) & np.isfinite(lons) & (np.abs(lats) <= 90) & (np.abs(lons) <= 180)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < len(df_avail):
        print(f"[osv5m] dropping {len(df_avail) - len(valid_idx)} rows with invalid lat/lon")

    sel_within_valid = stratified_indices_by_grid(
        lats[valid_idx], lons[valid_idx], args.n, bin_deg=args.bin_deg, seed=args.seed
    )
    sel = valid_idx[sel_within_valid]
    sel.sort()
    print(f"[osv5m] selected {len(sel)} rows from {len(df_avail)} available")
    df_sel = df_avail.iloc[sel].reset_index(drop=True)
    selected_ids = set(df_sel["_id_str"].tolist())

    print("[osv5m] extracting selected images ...")
    extracted: dict[str, Path] = {}
    for zp in downloaded_zips:
        with zipfile.ZipFile(zp) as zf:
            members = [n for n in zf.namelist() if Path(n).stem in selected_ids]
            for name in tqdm(members, desc=f"extract {zp.name}"):
                id_str = Path(name).stem
                if id_str in extracted:
                    continue
                target = images_dir / f"{id_str}.jpg"
                if not target.exists():
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                extracted[id_str] = target

    if not args.keep_extra:
        for p in images_dir.iterdir():
            if p.is_file() and p.stem not in selected_ids:
                p.unlink()

    records = []
    for _, row in df_sel.iterrows():
        id_str = row["_id_str"]
        target = images_dir / f"{id_str}.jpg"
        if not target.exists():
            continue
        rec = {
            "id": id_str,
            "lat": float(row[lat_col]),
            "lon": float(row[lon_col]),
            "path": str(target.relative_to(out_dir).as_posix()),
        }
        if country_col:
            rec["country"] = row.get(country_col)
        if region_col:
            rec["region"] = row.get(region_col)
        if sub_region_col:
            rec["sub_region"] = row.get(sub_region_col)
        records.append(rec)

    meta_out = pd.DataFrame.from_records(records)
    meta_out.to_csv(out_dir / "metadata.csv", index=False)
    print(f"[osv5m] wrote metadata to {out_dir / 'metadata.csv'}")
    print(f"[osv5m] done: {len(meta_out)} images in {images_dir}")


if __name__ == "__main__":
    main()
