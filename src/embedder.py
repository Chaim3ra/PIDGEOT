from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class StreetCLIPEmbedder:
    def __init__(self, model_name: str = "geolocal/StreetCLIP", device: torch.device | None = None):
        self.device = device or pick_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.use_fp16 = self.device.type == "cuda"
        if self.use_fp16:
            self.model = self.model.half()
        self.embed_dim = self.model.config.projection_dim

    @torch.inference_mode()
    def embed_pil(self, images: Iterable[Image.Image]) -> np.ndarray:
        pil_list: List[Image.Image] = [img.convert("RGB") for img in images]
        inputs = self.processor(images=pil_list, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.use_fp16:
            pixel_values = pixel_values.half()
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        pooled = (
            vision_outputs.pooler_output
            if hasattr(vision_outputs, "pooler_output")
            else vision_outputs[1]
        )
        feats = self.model.visual_projection(pooled)
        feats = F.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def embed_one(self, image: Image.Image) -> np.ndarray:
        return self.embed_pil([image])[0]
