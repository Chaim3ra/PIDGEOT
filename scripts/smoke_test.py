from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> None:
    print("[smoke]", " ".join(cmd))
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(f"[smoke] command failed: {' '.join(cmd)}")


def main() -> None:
    smoke_dir = ROOT / "data" / "smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    subset = smoke_dir / "osv5m_subset"
    cells_path = smoke_dir / "cells.pkl"
    emb_path = smoke_dir / "embeddings.npy"
    meta_path = smoke_dir / "meta.parquet"
    clf_path = smoke_dir / "classifier.pt"
    protos_path = smoke_dir / "prototypes.npz"

    python = sys.executable

    run([python, "scripts/download_osv5m.py", "--n", "1000", "--out", str(subset), "--split", "test", "--max-shards", "1", "--seed", "0"])
    run([python, "scripts/build_geocells.py", "--meta", str(subset / "metadata.csv"), "--out", str(cells_path), "--min-samples", "10"])
    run([python, "scripts/embed_dataset.py", "--meta", str(subset / "metadata.csv"), "--cells", str(cells_path), "--out-emb", str(emb_path), "--out-meta", str(meta_path)])
    run([python, "scripts/train_classifier.py", "--emb", str(emb_path), "--meta", str(meta_path), "--cells", str(cells_path), "--out", str(clf_path), "--curves", str(smoke_dir / "curves.png"), "--epochs", "1"])
    run([python, "scripts/build_prototypes.py", "--emb", str(emb_path), "--meta", str(meta_path), "--out", str(protos_path)])

    import pandas as pd
    from PIL import Image

    from src.pipeline import predict

    meta = pd.read_parquet(meta_path)
    val = meta[meta["split"] == "val"].head(1)
    if len(val) == 0:
        val = meta.head(1)
    img_path = subset / val.iloc[0]["path"]
    img = Image.open(img_path)
    result = predict(
        img,
        classifier_path=str(clf_path),
        prototypes_path=str(protos_path),
        top_cells=3,
        top_k=3,
    )
    print("[smoke] result:", result["predicted"], "conf", result["confidence"])
    assert -90 <= result["predicted"][0] <= 90, result
    assert -180 <= result["predicted"][1] <= 180, result
    assert len(result["top_k"]) > 0
    print("[smoke] OK")


if __name__ == "__main__":
    main()
