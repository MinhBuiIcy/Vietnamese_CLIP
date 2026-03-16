"""Training loop for a single Vietnamese CLIP pipeline."""

import argparse
import functools
import gc
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from .dataset import (
    EvalDataset,
    VietnameseCaptionDataset,
    collate_fn,
    load_hf_dataset,
)
from .evaluate import evaluate_retrieval
from .loss import clip_loss
from .models.clip_model import VietCLIP
from .models.image_encoder import ImageEncoder
from .models.text_encoder import TextEncoder, get_tokenizer
from .utils import (
    AverageMeter,
    Timer,
    cleanup_model,
    get_output_dir,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def build_model(image_enc_name: str, text_enc_name: str, embed_dim: int) -> VietCLIP:
    image_encoder = ImageEncoder(image_enc_name)
    text_encoder = TextEncoder(text_enc_name)
    model = VietCLIP(image_encoder, text_encoder, embed_dim=embed_dim)
    return model


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_pipeline(
    image_enc_name: str,
    text_enc_name: str,
    config: dict,
    output_base: str | Path = "outputs",
) -> dict:
    """
    Train a single (image_encoder, text_encoder) pipeline.

    Returns a metrics dict including timing information.
    """
    set_seed(config.get("seed", 42))
    timer = Timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = get_output_dir(output_base, image_enc_name, text_enc_name)

    logger.info("=== Pipeline: %s + %s ===", image_enc_name, text_enc_name)
    logger.info("Output dir: %s", output_dir)

    # ---- Data ----
    logger.info("Loading datasets...")
    train_hf = load_hf_dataset("train")
    val_hf = load_hf_dataset("validation")

    tokenizer = get_tokenizer(text_enc_name)

    # Build model first to get image transform
    logger.info("Building model...")
    model = build_model(image_enc_name, text_enc_name, config["embed_dim"])
    model = model.to(device)

    image_transform = model.image_encoder.get_transform()

    train_ds = VietnameseCaptionDataset(train_hf, transform=image_transform)
    val_ds = VietnameseCaptionDataset(val_hf, transform=image_transform)
    eval_ds = EvalDataset(val_hf)

    _collate = functools.partial(
        collate_fn, tokenizer=tokenizer, max_length=config["max_caption_length"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=True,
    )

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    epochs = config["epochs"]
    warmup_epochs = config.get("warmup_epochs", 2)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # BF16 AMP (preferred on Ampere/RTX 4090)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    scaler = GradScaler(enabled=False)  # BF16 doesn't need GradScaler

    # ---- Training ----
    best_mean_recall = -1.0
    best_epoch = 0
    grad_clip = config.get("grad_clip", 1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    img_emb, txt_emb, logit_scale = model(
                        pixel_values, input_ids, attention_mask
                    )
                    loss = clip_loss(img_emb, txt_emb, logit_scale)
            else:
                img_emb, txt_emb, logit_scale = model(
                    pixel_values, input_ids, attention_mask
                )
                loss = clip_loss(img_emb, txt_emb, logit_scale)

            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(model.trainable_parameters()), grad_clip
                )

            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), pixel_values.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | lr=%.2e",
            epoch,
            epochs,
            loss_meter.avg,
            scheduler.get_last_lr()[0],
        )

        # ---- Validation retrieval ----
        val_metrics = evaluate_retrieval(
            model,
            eval_ds,
            val_ds,
            tokenizer,
            device,
            batch_size=config["batch_size"],
            max_length=config["max_caption_length"],
        )

        mean_recall = val_metrics["mean_recall"]
        logger.info("Epoch %d | val mean_recall=%.2f%%", epoch, mean_recall)

        # Save best checkpoint
        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": {
                        "image_proj": model.image_proj.state_dict(),
                        "text_proj": model.text_proj.state_dict(),
                        "logit_scale": model.logit_scale.data,
                    },
                    "metrics": val_metrics,
                    "config": config,
                },
                output_dir / "best_model.pt",
            )
            logger.info("Saved best_model.pt (epoch %d)", epoch)

    # Save last checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": {
                "image_proj": model.image_proj.state_dict(),
                "text_proj": model.text_proj.state_dict(),
                "logit_scale": model.logit_scale.data,
            },
            "metrics": val_metrics,
            "config": config,
        },
        output_dir / "last_model.pt",
    )

    # ---- Final evaluation with best checkpoint ----
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.image_proj.load_state_dict(ckpt["model_state_dict"]["image_proj"])
    model.text_proj.load_state_dict(ckpt["model_state_dict"]["text_proj"])
    model.logit_scale.data = ckpt["model_state_dict"]["logit_scale"]

    final_metrics = evaluate_retrieval(
        model,
        eval_ds,
        val_ds,
        tokenizer,
        device,
        batch_size=config["batch_size"],
        max_length=config["max_caption_length"],
    )
    final_metrics["best_epoch"] = best_epoch
    final_metrics["total_train_time_min"] = round(timer.elapsed_minutes(), 2)
    final_metrics["image_encoder"] = image_enc_name
    final_metrics["text_encoder"] = text_enc_name

    save_json(final_metrics, output_dir / "metrics.json")
    logger.info("Pipeline complete. Metrics: %s", final_metrics)

    # Cleanup
    cleanup_model(model)

    return final_metrics


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Train a single Vietnamese CLIP pipeline")
    parser.add_argument("--image-encoder", required=True, help="Image encoder name")
    parser.add_argument("--text-encoder", required=True, help="Text encoder name")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--output-dir", default="outputs", help="Base output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs

    metrics = train_one_pipeline(
        args.image_encoder,
        args.text_encoder,
        config,
        output_base=args.output_dir,
    )
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
