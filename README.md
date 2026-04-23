# PIDGEOT — Image Geolocalization

A simplified re-implementation of [PIGEON](https://arxiv.org/abs/2307.05845) (Haas et al., CVPR 2024):

- **Pretrained [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP)** as the image encoder (no fine-tuning).
- **[OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m)** (Mapillary-derived) as the reference image source.
- **Semantic geocells** built by merging administrative boundaries (country → region → sub-region).
- **Haversine-smoothed cross-entropy** at the geocell head (PIGEON's core supervision trick, τ=75 km).
- **Prototype refinement**: within each top-K geocell, pick the nearest per-cluster prototype by cosine similarity to the query embedding.
- Simple **Gradio UI**: upload an image, see the predicted lat/lon and top-K candidates on a Folium world map.

## Pipeline

```
Query image
  → StreetCLIP ViT-L/14 (frozen, 768-d, L2-normalized)
  → MLP head (trained with haversine-smoothed CE) → top-K geocells
  → gather prototypes in those cells
  → cosine similarity → predicted lat/lon + top-K candidates
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
uv run scripts/download_osv5m.py --n 50000 --out data/osv5m_subset --split test --max-shards 2 --seed 42
uv run scripts/build_geocells.py --meta data/osv5m_subset/metadata.csv --out data/geocells/cells.pkl --min-samples 30
uv run scripts/embed_dataset.py --meta data/osv5m_subset/metadata.csv --cells data/geocells/cells.pkl --out-emb data/embeddings.npy --out-meta data/meta.parquet
uv run scripts/train_classifier.py --emb data/embeddings.npy --meta data/meta.parquet --cells data/geocells/cells.pkl --out data/classifier.pt --epochs 20 --tau-km 75
uv run scripts/build_prototypes.py --emb data/embeddings.npy --meta data/meta.parquet --out data/prototypes.npz
uv run scripts/eval.py --emb data/embeddings.npy --meta data/meta.parquet --classifier data/classifier.pt --protos data/prototypes.npz
uv run app.py
```

Open http://127.0.0.1:7860 in your browser.

### Haversine-smoothing ablation

Re-run the training with `--tau-km 0` to get the plain one-hot cross-entropy baseline and compare against the τ=75 run for the writeup:

```bash
uv run scripts/train_classifier.py --emb data/embeddings.npy --meta data/meta.parquet --cells data/geocells/cells.pkl --tau-km 0 --out data/classifier_onehot.pt --curves data/curves_onehot.png
```

## Layout

```
pidgeot/
├── app.py                          # Gradio UI
├── config.yaml                     # shared hyperparameters
├── requirements.txt
├── src/
│   ├── config.py                   # config loader
│   ├── data.py                     # subsampling + split helpers
│   ├── embedder.py                 # StreetCLIP wrapper
│   ├── geocells.py                 # semantic cells via admin hierarchy merging
│   ├── classifier.py               # MLP head + haversine-smoothing loss
│   ├── prototypes.py               # per-cell k-means prototypes
│   ├── pipeline.py                 # inference entrypoint used by app.py
│   ├── metrics.py                  # haversine, top-k acc, spherical centroid
│   └── viz.py                      # Folium map builder
├── scripts/
│   ├── download_osv5m.py
│   ├── build_geocells.py
│   ├── embed_dataset.py
│   ├── train_classifier.py
│   ├── build_prototypes.py
│   ├── eval.py
│   └── smoke_test.py
└── data/                           # git-ignored artifacts
```

## Key design choices

- **Semantic geocells.** Admin hierarchy (`country`, `region`, `sub_region`) from osv5m with `full=True`. Cells with fewer than `min_samples` (default 30) merge up into their parent; stragglers merge into the nearest same-continent neighbor by great-circle distance.
- **Loss.** Haversine-smoothed cross-entropy. For each sample with true lat/lon *x* and true cell centroid *g\**, the target over cells is `softmax_i( -(Hav(g_i, x) - Hav(g*, x)) / τ )` with τ=75 km. Monitored alongside top-1/top-5 cell accuracy and median haversine km at the cell-centroid level.
- **Prototype refinement.** For each geocell with ≥ 50 training points, run k-means (k up to 10) on embeddings to get representative prototypes; refinement picks argmax cosine similarity among prototypes drawn from the top-K predicted cells.

## Reference

Haas, L., Skreta, M., Alberti, S., & Finn, C. (2024). *PIGEON: Predicting Image Geolocations.* CVPR 2024. [arXiv:2307.05845](https://arxiv.org/abs/2307.05845)
