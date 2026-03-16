"""Text encoder wrappers using HuggingFace Transformers."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

TEXT_ENCODER_REGISTRY: dict[str, dict] = {
    "phobert": {
        "hf_id": "vinai/phobert-base",
        "output_dim": 768,
        "pooling": "cls",
    },
    "vibert": {
        "hf_id": "FPTAI/vibert-base-cased",
        "output_dim": 768,
        "pooling": "cls",
    },
    "vielect": {
        "hf_id": "FPTAI/vielect-base-discriminator",
        "output_dim": 768,
        "pooling": "cls",
    },
    "xlm_roberta": {
        "hf_id": "xlm-roberta-base",
        "output_dim": 768,
        "pooling": "cls",
    },
    "labse": {
        "hf_id": "sentence-transformers/LaBSE",
        "output_dim": 768,
        "pooling": "mean",
    },
}


class TextEncoder(nn.Module):
    """Frozen HuggingFace text encoder."""

    def __init__(self, name: str) -> None:
        super().__init__()
        cfg = TEXT_ENCODER_REGISTRY[name]
        self.model = AutoModel.from_pretrained(cfg["hf_id"])
        self.output_dim: int = cfg["output_dim"]
        self.pooling: str = cfg["pooling"]

        # Use hidden_size from config in case it differs
        if hasattr(self.model.config, "hidden_size"):
            self.output_dim = self.model.config.hidden_size

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0, :]
        else:  # mean pooling
            token_embeddings = outputs.last_hidden_state  # [B, T, D]
            mask = attention_mask.unsqueeze(-1).float()   # [B, T, 1]
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts

    def train(self, mode: bool = True):
        # Always keep the backbone in eval mode
        super().train(mode)
        self.model.eval()
        return self


def get_tokenizer(name: str) -> AutoTokenizer:
    """Return the tokenizer for a registered text encoder."""
    hf_id = TEXT_ENCODER_REGISTRY[name]["hf_id"]
    return AutoTokenizer.from_pretrained(hf_id, use_fast=True)
