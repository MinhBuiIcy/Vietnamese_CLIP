"""Dataset utilities for Vietnamese CLIP training."""

import logging
from collections import defaultdict
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class VietnameseCaptionDataset(Dataset):
    """
    Wraps the HuggingFace `ai-enthusiasm-community/coco-2017-vietnamese` dataset.

    Each item returns (image_tensor, caption_str, image_id).
    """

    def __init__(
        self,
        hf_dataset,
        transform: Callable | None = None,
    ) -> None:
        self.data = hf_dataset
        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        item = self.data[idx]
        image = item["image"]
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = self.transform(image)
        caption = item["caption_vi"]
        image_id = item["image_id"]
        return image_tensor, caption, image_id


def collate_fn(
    batch: list[tuple[torch.Tensor, str, int]],
    tokenizer,
    max_length: int = 128,
) -> dict[str, Any]:
    """
    Collate a batch of (image_tensor, caption, image_id) tuples.

    Tokenizes captions on-the-fly — encoder-agnostic.
    """
    images, captions, image_ids = zip(*batch)
    pixel_values = torch.stack(images)

    encoding = tokenizer(
        list(captions),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "pixel_values": pixel_values,
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "captions": list(captions),
        "image_ids": list(image_ids),
    }


class EvalDataset:
    """
    Groups validation samples by image_id for Recall@K evaluation.

    Attributes:
        unique_image_ids: sorted list of unique image IDs
        image_id_to_idx:  map from image_id to its index in unique_image_ids
        captions_per_image: {image_id: [caption, ...]}  (typically 5 per image)
        flat_captions: flat list of all captions in order
        flat_caption_image_ids: image_id for each entry in flat_captions
    """

    def __init__(self, hf_dataset) -> None:
        captions_by_id: dict[int, list[str]] = defaultdict(list)
        for item in hf_dataset:
            captions_by_id[item["image_id"]].append(item["caption_vi"])

        self.unique_image_ids: list[int] = sorted(captions_by_id.keys())
        self.image_id_to_idx: dict[int, int] = {
            iid: i for i, iid in enumerate(self.unique_image_ids)
        }
        self.captions_per_image: dict[int, list[str]] = dict(captions_by_id)

        self.flat_captions: list[str] = []
        self.flat_caption_image_ids: list[int] = []
        for iid in self.unique_image_ids:
            for cap in self.captions_per_image[iid]:
                self.flat_captions.append(cap)
                self.flat_caption_image_ids.append(iid)

        logger.info(
            "EvalDataset: %d unique images, %d total captions",
            len(self.unique_image_ids),
            len(self.flat_captions),
        )


def load_hf_dataset(split: str = "train"):
    """Load the HuggingFace COCO Vietnamese dataset for a given split."""
    from datasets import load_dataset

    ds = load_dataset(
        "ai-enthusiasm-community/coco-2017-vietnamese",
        split=split,
        trust_remote_code=True,
    )
    return ds
