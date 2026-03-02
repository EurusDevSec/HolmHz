"""
HolmHz Training Script — CLI entry point.

Usage:
    # Default config (configs/train.yaml)
    python scripts/train.py

    # Custom config
    python scripts/train.py configs/train.yaml

    # Override specific values
    python scripts/train.py --batch_size 16 --epochs 5

Example dry run (local, 2 epochs):
    python scripts/train.py --epochs 2 --num_workers 0 --batch_size 8

Full training (Kaggle GPU):
    python scripts/train.py  # Uses default config
"""

import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Load .env (WANDB_API_KEY, etc.) before any wandb calls
load_dotenv()

# ─── Import HolmHz modules ───
import holmhz.detectors  # Trigger DETECTOR_REGISTRY registration  # noqa: F401
from holmhz.data import create_dataloader
from holmhz.losses import get_loss_fn
from holmhz.training import EarlyStopping, Trainer, get_scheduler
from holmhz.utils.logger import get_logger
from holmhz.utils.registry import DETECTOR_REGISTRY

logger = get_logger("train")


def main():
    """Main training entry point."""
    # ─── Load config ───
    config_path = "configs/train.yaml"

    # Detect if first arg is a YAML config file (not an override like key=value)
    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("--")
        and "=" not in sys.argv[1]
    ):
        config_path = sys.argv[1]

    config = OmegaConf.load(config_path)

    # CLI overrides (key=value or --key value)
    cli_args = [a for a in sys.argv[1:] if a != config_path]
    if cli_args:
        cli_overrides = OmegaConf.from_cli(cli_args)
        config = OmegaConf.merge(config, cli_overrides)

    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # ─── Device ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ─── Data ───
    use_sampler = OmegaConf.select(config, "data.use_weighted_sampler", default=False)
    train_loader = create_dataloader(
        manifest_path=config.data.train_manifest,
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        is_training=True,
        num_workers=config.data.num_workers,
        use_weighted_sampler=use_sampler,
    )
    val_loader = create_dataloader(
        manifest_path=config.data.val_manifest,
        batch_size=config.training.batch_size * 2,  # Val batch lớn hơn (no gradient)
        image_size=config.data.image_size,
        is_training=False,
        num_workers=config.data.num_workers,
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    if use_sampler:
        logger.info("WeightedRandomSampler: ENABLED (balanced source sampling)")

    # ─── Model ───
    model = DETECTOR_REGISTRY.build(
        config.model.name,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    # ─── Optimizer ───
    # Chỉ optimize params có requires_grad (head khi freeze backbone)
    params = [p for p in model.parameters() if p.requires_grad]

    if config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=config.training.learning_rate,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    # ─── Scheduler ───
    scheduler = get_scheduler(
        optimizer,
        name=config.training.scheduler,
        epochs=config.training.epochs,
    )

    # ─── Loss ───
    # pos_weight: tăng penalty khi miss class positive (Fake)
    # Hữu ích khi sources có số lượng khác nhau
    pos_weight_val = None
    if hasattr(config, "training") and hasattr(config.training, "pos_weight"):
        pos_weight_val = config.training.pos_weight
        logger.info(f"pos_weight: {pos_weight_val}")

    loss_fn = get_loss_fn("bce_with_logits", pos_weight=pos_weight_val)
    if pos_weight_val is not None:
        # Move pos_weight to same device as model
        loss_fn = loss_fn.to(device)

    # ─── Early Stopping ───
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping.patience,
        mode="max",  # val_auc — higher is better
    )

    # ─── W&B ───
    use_wandb = False
    try:
        import wandb

        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.get("entity"),
            config=OmegaConf.to_container(config, resolve=True),
        )
        use_wandb = True
        logger.info(f"W&B run: {wandb.run.name}")
    except Exception as e:
        logger.warning(f"W&B disabled: {e}")

    # ─── Trainer ───
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        early_stopping=early_stopping,
        config=OmegaConf.to_container(config, resolve=True),
        device=device,
        use_wandb=use_wandb,
    )

    # ─── Resume from checkpoint ───
    resume_path = Path("outputs/checkpoints/last.pt")
    if resume_path.exists():
        trainer.load_checkpoint(str(resume_path))

    # ─── Train ───
    trainer.fit(config.training.epochs)

    # ─── Cleanup ───
    if use_wandb:
        import wandb

        wandb.finish()

    logger.info("Done!")


if __name__ == "__main__":
    main()
