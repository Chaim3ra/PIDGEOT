# PIDGEOT: Adding Classical Vision To Image Geolocalization

A simplified re-implementation of [PIGEON](https://arxiv.org/abs/2307.05845) (Haas et al., CVPR 2024):

- **Pretrained [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP)** as the image encoder (no fine-tuning).
- **[OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m)** (Mapillary-derived) as the reference image source.
- **Semantic geocells** built by merging administrative boundaries (country → region → sub-region).
- **Haversine-smoothed cross-entropy** at the geocell head (PIGEON's core supervision trick, τ=75 km).
- **Prototype refinement**: within each top-K geocell, pick the nearest per-cluster prototype by cosine similarity to the query embedding.
- **OCR-script re-ranking** (β term): EasyOCR → Unicode script classification → per-country accepted-script table → asymmetric bonus.
- **Color-histogram re-ranking** (γ term): HSV 3×3 spatial-grid histograms, intersection similarity, per-prototype mean histogram.
- Simple **Gradio UI**: upload an image, toggle each re-rank term live, see the top-K on a Folium world map.

## Group Members
- Chaitanya Agarwal (agarwal.cha@northeastern.edu)
- Lohith Chamakura (chamakura.l@northeastern.edu)
- Rishabh Kumar (kumar.rishabh@northeastern.edu)

## Project Demo (Video Link)
https://drive.google.com/file/d/1vIYBHqk_ZuBbu-4WM_Vw13ikToSVXPPK/view?usp=drive_link



## Pipeline

```
Query image
  → StreetCLIP ViT-L/14 (frozen, 768-d, L2-normalized)
  → MLP head (trained with haversine-smoothed CE) → top-K geocells
  → gather prototypes in those cells
  →  α · clip_cos_sim   (prototype CLIP similarity)
   + β · script_bonus   (OCR detection vs. candidate country scripts)
   + γ · color_sim      (HSV histogram intersection vs. candidate prototype)
  → predicted lat/lon + top-K candidates
```

## Setup

This project uses [**uv**](https://docs.astral.sh/uv/) to manage the Python environment. Install uv first ([install guide](https://docs.astral.sh/uv/getting-started/installation/)), then:

```bash
uv venv                                      # create .venv with a compatible Python
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision   # GPU build
uv pip install -r requirements.txt
```

On CPU-only machines skip the CUDA line — the plain `uv pip install -r requirements.txt` pulls the default CPU `torch` wheel.

All scripts below are run via `uv run`, which auto-activates `.venv` — no manual `source .venv/bin/activate` needed.

## End-to-end

```bash
uv run scripts/smoke_test.py                                                # 1k-image dry run (optional)

# 1. Data: 50k-image stratified subset of osv5m.
uv run scripts/download_osv5m.py --n 50000 --out data/osv5m_subset --split test --max-shards 2 --seed 42

# 2. Semantic geocells from admin hierarchy.
uv run scripts/build_geocells.py --meta data/osv5m_subset/metadata.csv --out data/geocells/cells.pkl --min-samples 30

# 3. StreetCLIP embedding pass (GPU-heavy; ~30 min on a modern GPU).
uv run scripts/embed_dataset.py --meta data/osv5m_subset/metadata.csv --cells data/geocells/cells.pkl --out-emb data/embeddings.npy --out-meta data/meta.parquet

# 4. MLP head with haversine-smoothed CE.
uv run scripts/train_classifier.py --emb data/embeddings.npy --meta data/meta.parquet --cells data/geocells/cells.pkl --out data/classifier.pt --epochs 20 --tau-km 75

# 5. Per-image HSV histograms (for color re-ranking).
uv run scripts/compute_histograms.py --meta data/meta.parquet --out data/image_histograms.npy

# 6. Per-geocell prototypes (embedding means + country labels + histogram means).
uv run scripts/build_prototypes.py --emb data/embeddings.npy --meta data/meta.parquet --cells data/geocells/cells.pkl --image-histograms data/image_histograms.npy --out data/prototypes.npz

# 7. Grid-search rerank weights β (OCR) and γ (color). Caches OCR on first run
#    (~1-2 h on GPU); subsequent grid-searches finish in seconds. Writes the
#    best (β, γ) back into config.yaml.
uv run scripts/tune_rerank.py --meta data/meta.parquet --emb data/embeddings.npy --cells data/geocells/cells.pkl --classifier data/classifier.pt --protos data/prototypes.npz --cache data/ocr_cache.parquet --image-histograms data/image_histograms.npy --results-dir results --update-config

# 8. Full-val + per-subset evaluation (uses β, γ from config).
uv run scripts/eval.py --emb data/embeddings.npy --meta data/meta.parquet --classifier data/classifier.pt --protos data/prototypes.npz

# 9. Launch the Gradio demo (toggle each rerank term + slide β/γ live).
uv run app.py
```

Open http://127.0.0.1:7860 in your browser.

### Haversine-smoothing ablation

Re-run the training with `--tau-km 0` to get the plain one-hot cross-entropy baseline and compare against the τ=75 run for the writeup:

```bash
uv run scripts/train_classifier.py --emb data/embeddings.npy --meta data/meta.parquet --cells data/geocells/cells.pkl --tau-km 0 --out data/classifier_onehot.pt --curves results/curves_onehot.png
```

## Layout

```
pidgeot/
├── app.py                          # Gradio UI (toggles + sliders for each rerank term)
├── config.yaml                     # shared hyperparameters
├── requirements.txt
├── docs/
│   └── IMPLEMENTATION.md           # design walkthrough (why each choice)
├── src/
│   ├── config.py                   # config loader
│   ├── data.py                     # subsampling + split helpers
│   ├── embedder.py                 # StreetCLIP wrapper (ViT-L/14, fp16 on CUDA)
│   ├── geocells.py                 # semantic cells via admin-hierarchy merging
│   ├── classifier.py               # MLP head + haversine-smoothed CE loss
│   ├── prototypes.py               # per-cell k-means prototypes (emb + country + hist)
│   ├── ocr.py                      # multi-reader EasyOCR ScriptDetector
│   ├── country_scripts.py          # country → accepted-scripts table + bonus rule
│   ├── color_hist.py               # HSV 3×3 grid histogram + intersection sim
│   ├── pipeline.py                 # inference entry used by app.py (α + β + γ scoring)
│   ├── metrics.py                  # haversine, top-k acc, spherical centroid
│   └── viz.py                      # Folium map builder
├── scripts/
│   ├── download_osv5m.py
│   ├── build_geocells.py
│   ├── embed_dataset.py
│   ├── train_classifier.py
│   ├── compute_histograms.py
│   ├── build_prototypes.py
│   ├── tune_rerank.py              # β, γ, and β×γ grid search
│   ├── eval.py
│   └── smoke_test.py
├── data/                           # git-ignored artifacts (images, embeddings, caches)
└── results/                        # git-tracked charts + metrics JSON
```

## Key design choices

- **Semantic geocells.** Admin hierarchy (`country`, `region`, `sub-region`) ships directly in osv5m's `train.csv` / `test.csv`. Groups with fewer than `min_samples` (default 30) merge up into their parent; orphan countries merge into the nearest same-continent neighbor by great-circle distance.
- **Loss.** Haversine-smoothed cross-entropy. For each sample with true lat/lon *x* and true cell centroid *g\**, the target over cells is `softmax_i( -(Hav(g_i, x) - Hav(g*, x)) / τ )` with τ=75 km. Monitored alongside top-1/top-5 cell accuracy and median haversine km at the cell-centroid level.
- **Prototype refinement.** For each geocell, run k-means on the cell's training embeddings (k up to 10, scales with cell size). Refinement picks argmax CLIP cos-sim among prototypes drawn from the top-K predicted cells.
- **OCR re-rank (β).** 7 EasyOCR readers (one per non-Latin script family + `en` bundled), IoU-deduped detections, Unicode-range script buckets, asymmetric bonus (+1 if any detected non-Latin script matches the country's accepted set, −1 if not, 0 if only Latin was detected).
- **Color re-rank (γ).** HSV 8×8×8 histograms on a 3×3 spatial grid, L1-normalized per cell, concatenated to a 4608-d flat vector. Per-prototype mean histogram. Histogram-intersection similarity averaged across the 9 cells.

## Reference

Haas, L., Skreta, M., Alberti, S., & Finn, C. (2024). *PIGEON: Predicting Image Geolocations.* CVPR 2024. [arXiv:2307.05845](https://arxiv.org/abs/2307.05845)
