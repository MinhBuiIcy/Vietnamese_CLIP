"""Image encoder wrappers using timm."""

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config

IMAGE_ENCODER_REGISTRY: dict[str, dict] = {
    "resnet50": {
        "timm_id": "resnet50",
        "output_dim": 2048,
    },
    "efficientnet_b0": {
        "timm_id": "efficientnet_b0",
        "output_dim": 1280,
    },
    "convnext_tiny": {
        "timm_id": "convnext_tiny.fb_in22k_ft_in1k",
        "output_dim": 768,
    },
    "mobilenet_v3": {
        "timm_id": "mobilenetv3_large_100",
        "output_dim": 960,
    },
}


class ImageEncoder(nn.Module):
    """Frozen timm image encoder."""

    def __init__(self, name: str) -> None:
        super().__init__()
        cfg = IMAGE_ENCODER_REGISTRY[name]
        self.model = timm.create_model(
            cfg["timm_id"],
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.output_dim: int = cfg["output_dim"]

        # Resolve data config for correct preprocessing
        data_cfg = resolve_model_data_config(self.model)
        self._transform = create_transform(**data_cfg, is_training=False)

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_transform(self):
        """Return the inference transform for this encoder."""
        return self._transform

    def train(self, mode: bool = True):
        # Always keep the backbone in eval mode (frozen BN etc.)
        super().train(mode)
        self.model.eval()
        return self
