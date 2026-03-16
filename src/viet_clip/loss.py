"""Symmetric InfoNCE loss for CLIP-style training."""

import torch
import torch.nn.functional as F
from torch import Tensor


def clip_loss(image_embeds: Tensor, text_embeds: Tensor, logit_scale: Tensor) -> Tensor:
    """
    Symmetric InfoNCE (contrastive) loss.

    Args:
        image_embeds: L2-normalized image embeddings [B, D]
        text_embeds:  L2-normalized text  embeddings [B, D]
        logit_scale:  Learnable scalar (already exponentiated or raw log-scale)

    Returns:
        Scalar loss value.
    """
    # logit_scale is stored as log(scale) and clamped; exponentiate before use
    scale = logit_scale.exp()
    logits = scale * image_embeds @ text_embeds.T  # [B, B]

    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)
