"""Utility functions for Vietnamese CLIP training."""

import gc
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_output_dir(base_dir: str | Path, image_enc: str, text_enc: str) -> Path:
    out = Path(base_dir) / f"{image_enc}__{text_enc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level.upper()),
    )


def cleanup_model(model: Any) -> None:
    """Delete model and free GPU memory."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class AverageMeter:
    """Tracks running average of a metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    def __init__(self) -> None:
        self._start = time.time()

    def elapsed_minutes(self) -> float:
        return (time.time() - self._start) / 60.0
