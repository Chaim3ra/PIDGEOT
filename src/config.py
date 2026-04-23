from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelCfg:
    name: str
    embed_dim: int
    image_size: int


@dataclass
class DataCfg:
    subset_dir: str
    metadata_csv: str
    images_dir: str
    n_samples: int
    val_fraction: float


@dataclass
class GeocellsCfg:
    path: str
    min_samples_per_cell: int
    target_num_cells: int


@dataclass
class EmbeddingsCfg:
    path: str
    meta_path: str
    batch_size: int
    batch_size_cpu: int


@dataclass
class ClassifierCfg:
    path: str
    hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    tau_km: float


@dataclass
class PrototypesCfg:
    path: str
    max_k_per_cell: int
    samples_per_prototype: int


@dataclass
class InferenceCfg:
    top_cells: int
    top_k_candidates: int


@dataclass
class Config:
    seed: int
    model: ModelCfg
    data: DataCfg
    geocells: GeocellsCfg
    embeddings: EmbeddingsCfg
    classifier: ClassifierCfg
    prototypes: PrototypesCfg
    inference: InferenceCfg


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        seed=raw["seed"],
        model=ModelCfg(**raw["model"]),
        data=DataCfg(**raw["data"]),
        geocells=GeocellsCfg(**raw["geocells"]),
        embeddings=EmbeddingsCfg(**raw["embeddings"]),
        classifier=ClassifierCfg(**raw["classifier"]),
        prototypes=PrototypesCfg(**raw["prototypes"]),
        inference=InferenceCfg(**raw["inference"]),
    )
