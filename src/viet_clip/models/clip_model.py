"""VietCLIP model: frozen encoders + trainable projection heads."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class ProjectionHead(nn.Module):
    """Linear → LayerNorm → ReLU → Linear projection."""

    def __init__(self, input_dim: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VietCLIP(nn.Module):
    """
    Vietnamese CLIP model.

    Frozen encoders + learnable projection heads + learnable temperature.
    Only image_proj, text_proj, and logit_scale have requires_grad=True.
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.image_proj = ProjectionHead(image_encoder.output_dim, embed_dim)
        self.text_proj = ProjectionHead(text_encoder.output_dim, embed_dim)

        # Learnable temperature: log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07)
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.image_encoder(pixel_values)
        embeddings = self.image_proj(features)
        return F.normalize(embeddings, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            features = self.text_encoder(input_ids, attention_mask)
        embeddings = self.text_proj(features)
        return F.normalize(embeddings, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Clamp temperature before use
        self.logit_scale.data.clamp_(max=math.log(100.0))

        img_emb = self.encode_image(pixel_values)
        txt_emb = self.encode_text(input_ids, attention_mask)
        return img_emb, txt_emb, self.logit_scale

    def trainable_parameters(self):
        """Yield only the parameters that should be optimized."""
        yield from self.image_proj.parameters()
        yield from self.text_proj.parameters()
        yield self.logit_scale

    def train(self, mode: bool = True):
        super().train(mode)
        # Re-apply eval to frozen encoders to keep BN in eval mode
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self
