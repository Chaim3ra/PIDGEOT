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
    hist_path = smoke_dir / "image_histograms.npy"

    python = sys.executable

    run([python, "scripts/download_osv5m.py", "--n", "1000", "--out", str(subset), "--split", "test", "--max-shards", "1", "--seed", "0"])
    run([python, "scripts/build_geocells.py", "--meta", str(subset / "metadata.csv"), "--out", str(cells_path), "--min-samples", "10"])
    run([python, "scripts/embed_dataset.py", "--meta", str(subset / "metadata.csv"), "--cells", str(cells_path), "--out-emb", str(emb_path), "--out-meta", str(meta_path)])
    run([python, "scripts/train_classifier.py", "--emb", str(emb_path), "--meta", str(meta_path), "--cells", str(cells_path), "--out", str(clf_path), "--curves", str(smoke_dir / "curves.png"), "--epochs", "1"])
    run([python, "scripts/compute_histograms.py", "--meta", str(meta_path), "--out", str(hist_path), "--images-root", str(subset)])
    run([python, "scripts/build_prototypes.py", "--emb", str(emb_path), "--meta", str(meta_path), "--cells", str(cells_path), "--image-histograms", str(hist_path), "--out", str(protos_path)])

    import numpy as np
    import pandas as pd
    from PIL import Image

    from src.ocr import ScriptDetector
    from src.pipeline import predict

    # 1. Pipeline smoke (CLIP-only path — skip OCR to keep the smoke run fast,
    # but exercise the color-hist path since it's cheap.)
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
        enable_ocr=False,
        enable_color=True,
        gamma=0.3,
    )
    print("[smoke] result:", result["predicted"], "conf", result["confidence"])
    assert -90 <= result["predicted"][0] <= 90, result
    assert -180 <= result["predicted"][1] <= 180, result
    assert len(result["top_k"]) > 0
    assert result["ocr_enabled"] is False
    assert result["detected_scripts"] is None
    assert result["color_enabled"] is True, result
    assert result["top_k"][0]["color_sim"] is not None, result["top_k"][0]

    # 2. Prototype country + hist columns populated
    protos = np.load(protos_path)
    assert "country" in protos.files, "prototypes.npz missing 'country' column"
    assert "hist" in protos.files, "prototypes.npz missing 'hist' column"
    assert "hist_bins" in protos.files and "hist_grid" in protos.files
    print(
        f"[smoke] prototypes: country n={len(protos['country'])}, "
        f"hist shape={protos['hist'].shape}, bins={protos['hist_bins'].tolist()}, "
        f"grid={int(protos['hist_grid'][0])}"
    )

    # 3. Per-image histogram cache is aligned with meta
    image_hists = np.load(hist_path)
    assert len(image_hists) == len(meta), (len(image_hists), len(meta))
    assert image_hists.shape[1] == protos["hist"].shape[1]
    print(f"[smoke] image_histograms shape={image_hists.shape}")

    # 4. OCR schema smoke (single image; verifies easyocr import + detect shape)
    detector = ScriptDetector.get()
    det = detector.detect(img)
    assert set(det.keys()) == {"scripts", "text", "dominant"}, det.keys()
    assert isinstance(det["scripts"], dict)
    assert isinstance(det["text"], list)
    print(f"[smoke] OCR schema OK; dominant={det['dominant']} scripts={det['scripts']}")

    print("[smoke] OK")


if __name__ == "__main__":
    main()
