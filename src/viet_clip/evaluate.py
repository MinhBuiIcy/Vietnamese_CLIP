"""Retrieval evaluation (Recall@K) for Vietnamese CLIP."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import EvalDataset, VietnameseCaptionDataset, collate_fn
from .models.clip_model import VietCLIP

logger = logging.getLogger(__name__)


@torch.no_grad()
def build_embeddings(
    model: VietCLIP,
    eval_ds: EvalDataset,
    image_dataset: VietnameseCaptionDataset,
    tokenizer,
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 128,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build image and text embedding matrices.

    Returns:
        image_embeddings: [N, D] for N unique images
        text_embeddings:  [M, D] for M total captions (M = 5*N typically)
    """
    model.eval()

    # --- Image embeddings: one per unique image ---
    # Read image_id column directly from Arrow (no image decoding)
    all_image_ids = image_dataset.data["image_id"]
    image_id_to_dataset_idx: dict[int, int] = {}
    for idx, iid in enumerate(all_image_ids):
        if iid not in image_id_to_dataset_idx:
            image_id_to_dataset_idx[iid] = idx

    # Use eval_ds ordering
    unique_samples = [
        image_id_to_dataset_idx[iid]
        for iid in eval_ds.unique_image_ids
        if iid in image_id_to_dataset_idx
    ]

    image_embs = []
    for start in range(0, len(unique_samples), batch_size):
        batch_idx = unique_samples[start : start + batch_size]
        imgs = torch.stack([image_dataset[i][0] for i in batch_idx]).to(device)
        emb = model.encode_image(imgs)
        image_embs.append(emb.cpu().float().numpy())
    image_embeddings = np.concatenate(image_embs, axis=0)  # [N, D]

    # --- Text embeddings: all captions in flat order ---
    text_embs = []
    captions = eval_ds.flat_captions
    for start in range(0, len(captions), batch_size):
        batch_caps = captions[start : start + batch_size]
        enc = tokenizer(
            batch_caps,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        emb = model.encode_text(input_ids, attention_mask)
        text_embs.append(emb.cpu().float().numpy())
    text_embeddings = np.concatenate(text_embs, axis=0)  # [M, D]

    return image_embeddings, text_embeddings


@torch.no_grad()
def evaluate_retrieval(
    model: VietCLIP,
    eval_ds: EvalDataset,
    image_dataset: VietnameseCaptionDataset,
    tokenizer,
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 128,
) -> dict[str, float]:
    """
    Compute image-to-text and text-to-image Recall@1/5/10.

    Returns dict with keys:
        i2t_R@1, i2t_R@5, i2t_R@10
        t2i_R@1, t2i_R@5, t2i_R@10
        mean_recall
    """
    image_embeddings, text_embeddings = build_embeddings(
        model, eval_ds, image_dataset, tokenizer, device, batch_size, max_length
    )

    N = len(eval_ds.unique_image_ids)
    M = len(eval_ds.flat_captions)

    # Precompute captions-per-image count for each image
    # and flat index ranges
    caption_idx_for_image: dict[int, list[int]] = {}
    for flat_idx, iid in enumerate(eval_ds.flat_caption_image_ids):
        img_idx = eval_ds.image_id_to_idx[iid]
        caption_idx_for_image.setdefault(img_idx, []).append(flat_idx)

    # Similarity matrix [N, M]
    sims = image_embeddings @ text_embeddings.T  # [N, M]

    # --- Image-to-text retrieval ---
    i2t_hits = {1: 0, 5: 0, 10: 0}
    for img_idx in range(N):
        ranked = np.argsort(-sims[img_idx])  # descending
        gt_caps = set(caption_idx_for_image.get(img_idx, []))
        for K in (1, 5, 10):
            top_k = set(ranked[:K].tolist())
            if top_k & gt_caps:
                i2t_hits[K] += 1

    # --- Text-to-image retrieval ---
    sims_t2i = sims.T  # [M, N]
    t2i_hits = {1: 0, 5: 0, 10: 0}
    for cap_idx in range(M):
        iid = eval_ds.flat_caption_image_ids[cap_idx]
        gt_img_idx = eval_ds.image_id_to_idx[iid]
        ranked = np.argsort(-sims_t2i[cap_idx])
        for K in (1, 5, 10):
            if gt_img_idx in ranked[:K].tolist():
                t2i_hits[K] += 1

    i2t_r1 = 100.0 * i2t_hits[1] / N
    i2t_r5 = 100.0 * i2t_hits[5] / N
    i2t_r10 = 100.0 * i2t_hits[10] / N

    t2i_r1 = 100.0 * t2i_hits[1] / M
    t2i_r5 = 100.0 * t2i_hits[5] / M
    t2i_r10 = 100.0 * t2i_hits[10] / M

    mean_recall = (i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10) / 6.0

    metrics = {
        "i2t_R@1": round(i2t_r1, 4),
        "i2t_R@5": round(i2t_r5, 4),
        "i2t_R@10": round(i2t_r10, 4),
        "t2i_R@1": round(t2i_r1, 4),
        "t2i_R@5": round(t2i_r5, 4),
        "t2i_R@10": round(t2i_r10, 4),
        "mean_recall": round(mean_recall, 4),
    }
    logger.info("Retrieval metrics: %s", metrics)
    return metrics
