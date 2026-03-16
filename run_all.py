"""
Orchestrator: run all 8 Vietnamese CLIP ablation pipelines sequentially.

Group 1 — Fix PhoBERT, vary image encoder (A–D)
Group 2 — Fix EfficientNet-B0, vary text encoder (E/B, F–I)
  (B and E share the same checkpoint: efficientnet_b0 + phobert)
"""

import csv
import logging
import traceback
from datetime import datetime
from pathlib import Path

from src.viet_clip.train import train_one_pipeline
from src.viet_clip.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

# 8 unique pipelines (experiment_id → (image_enc, text_enc))
EXPERIMENTS: list[tuple[str, str, str]] = [
    # (experiment_id, image_enc, text_enc)
    ("A", "resnet50",        "phobert"),
    ("B", "efficientnet_b0", "phobert"),    # shared baseline (also E)
    ("C", "convnext_tiny",   "phobert"),
    ("D", "mobilenet_v3",    "phobert"),
    # Group 2: vary text encoder with EfficientNet-B0
    # B (efficientnet_b0 + phobert) already covered above
    ("F", "efficientnet_b0", "vibert"),
    ("G", "efficientnet_b0", "vielect"),
    ("H", "efficientnet_b0", "xlm_roberta"),
    ("I", "efficientnet_b0", "labse"),
]

CSV_COLUMNS = [
    "experiment_id",
    "image_encoder",
    "text_encoder",
    "timestamp",
    "i2t_R@1",
    "i2t_R@5",
    "i2t_R@10",
    "t2i_R@1",
    "t2i_R@5",
    "t2i_R@10",
    "mean_recall",
    "best_epoch",
    "total_train_time_min",
    "error",
]


def append_to_csv(row: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Results appended to %s", csv_path)


def main() -> None:
    setup_logging()

    config = load_config("configs/default.yaml")
    output_base = Path("outputs")
    csv_path = output_base / "all_results.csv"

    logger.info("Starting full ablation study: %d pipelines", len(EXPERIMENTS))

    for exp_id, img_enc, txt_enc in EXPERIMENTS:
        logger.info("=" * 60)
        logger.info("Experiment %s: %s + %s", exp_id, img_enc, txt_enc)
        logger.info("=" * 60)

        row: dict = {
            "experiment_id": exp_id,
            "image_encoder": img_enc,
            "text_encoder": txt_enc,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "error": "",
        }

        try:
            metrics = train_one_pipeline(
                img_enc, txt_enc, config, output_base=output_base
            )
            row.update(
                {
                    "i2t_R@1": metrics.get("i2t_R@1", ""),
                    "i2t_R@5": metrics.get("i2t_R@5", ""),
                    "i2t_R@10": metrics.get("i2t_R@10", ""),
                    "t2i_R@1": metrics.get("t2i_R@1", ""),
                    "t2i_R@5": metrics.get("t2i_R@5", ""),
                    "t2i_R@10": metrics.get("t2i_R@10", ""),
                    "mean_recall": metrics.get("mean_recall", ""),
                    "best_epoch": metrics.get("best_epoch", ""),
                    "total_train_time_min": metrics.get("total_train_time_min", ""),
                }
            )
        except Exception as exc:
            logger.error(
                "Experiment %s FAILED: %s", exp_id, exc, exc_info=True
            )
            row["error"] = traceback.format_exc(limit=3)

        append_to_csv(row, csv_path)

    logger.info("All experiments complete. Results saved to %s", csv_path)


if __name__ == "__main__":
    main()
