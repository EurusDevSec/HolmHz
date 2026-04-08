"""
Trainer class — Orchestrate toàn bộ training pipeline.

Trainer là "nhạc trưởng" điều phối:
  DataLoader → Model → Loss → Optimizer → Metrics → Logging → Checkpoint

Pattern từ:
- DeepfakeBench: trainer/trainer.py (base class phức tạp)
- CNNDetection: train.py (script đơn giản, không class)
- HolmHz: Lấy ý tưởng DeepfakeBench nhưng ĐƠN GIẢN HÓA

Flow mỗi epoch:
  1. train_one_epoch() → iterate train_loader, compute loss, backward
  2. validate()        → iterate val_loader, compute metrics (no gradient)
  3. scheduler.step()  → giảm learning rate
  4. early_stopping()  → kiểm tra val_auc có cải thiện không
  5. save_checkpoint() → lưu best.pt và last.pt
  6. wandb.log()       → log metrics lên dashboard

Checkpoint resume:
  - Nếu last.pt tồn tại → tự động resume từ epoch tiếp theo
  - Quan trọng cho Kaggle/Colab bị disconnect giữa chừng
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..metrics.accuracy import compute_accuracy
from ..metrics.auc import compute_auc
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Orchestrate training: train loop, validation, logging, checkpoint.

    Args:
        model: detector model (từ Task 1.4)
        train_loader: training DataLoader (từ Task 1.3)
        val_loader: validation DataLoader (từ Task 1.3)
        optimizer: PyTorch optimizer (AdamW)
        scheduler: LR scheduler (CosineAnnealing)
        loss_fn: loss function (BCEWithLogitsLoss)
        early_stopping: EarlyStopping instance
        config: dict config (từ OmegaConf)
        device: torch.device (cuda hoặc cpu)
        checkpoint_dir: thư mục lưu checkpoint
        use_wandb: bật/tắt W&B logging
        use_amp: bật/tắt Mixed Precision (AMP)

    Example:
        >>> trainer = Trainer(model, train_loader, val_loader, ...)
        >>> trainer.fit(epochs=30)
        # → Train 30 epochs, save best.pt + last.pt
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        early_stopping,
        config: dict,
        device: torch.device,
        checkpoint_dir: str = "outputs/checkpoints",
        use_wandb: bool = True,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.early_stopping = early_stopping
        self.config = config
        self.device = device
        self.use_wandb = use_wandb

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed Precision — chỉ bật trên CUDA
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # State tracking
        self.start_epoch = 0
        self.best_metric = 0.0
        self.history: list[dict] = []

    # ──────────────────────────────────────────────────────────
    # TRAIN ONE EPOCH
    # ──────────────────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> dict:
        """Train model trên toàn bộ training set (1 epoch).

        Flow mỗi batch:
          1. Load images + labels → GPU
          2. Forward pass (AMP autocast)
          3. Compute loss
          4. Backward pass (GradScaler)
          5. Optimizer step
          6. Accumulate metrics

        Args:
            epoch: epoch index (0-based)

        Returns:
            dict với train_loss, train_acc, train_auc
        """
        self.model.train()  # Bật Dropout, BatchNorm training mode
        total_loss = 0.0
        all_logits = []
        all_labels = []
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            leave=False,
        )

        for batch in pbar:
            # Chuyển data lên device (GPU/CPU)
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # ─── Forward pass (Mixed Precision) ───
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                logits = self.model(images)                # [B, 1]
                loss = self.loss_fn(logits.squeeze(1), labels)  # [B]

            # ─── Backward pass (GradScaler) ───
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ─── Accumulate ───
            total_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())
            num_batches += 1

            # Progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Epoch-level metrics
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        return {
            "train_loss": total_loss / max(num_batches, 1),
            "train_acc": compute_accuracy(all_logits, all_labels),
            "train_auc": compute_auc(all_logits, all_labels),
        }

    # ──────────────────────────────────────────────────────────
    # VALIDATE
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Evaluate model trên validation set (không tính gradient).

        @torch.no_grad() = tắt gradient tracking → tiết kiệm VRAM + nhanh hơn.

        Args:
            epoch: epoch index (0-based)

        Returns:
            dict với val_loss, val_acc, val_auc
        """
        self.model.eval()  # Tắt Dropout, dùng running stats BatchNorm
        total_loss = 0.0
        all_logits = []
        all_labels = []
        num_batches = 0

        for batch in tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1} [Val]",
            leave=False,
        ):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            logits = self.model(images)                      # [B, 1]
            loss = self.loss_fn(logits.squeeze(1), labels)   # scalar

            total_loss += loss.item()
            all_logits.append(logits)
            all_labels.append(labels)
            num_batches += 1

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_acc": compute_accuracy(all_logits, all_labels),
            "val_auc": compute_auc(all_logits, all_labels),
        }

    # ──────────────────────────────────────────────────────────
    # FIT (main training loop)
    # ──────────────────────────────────────────────────────────

    def fit(self, epochs: int) -> list[dict]:
        """Main training loop — chạy train + validate cho mỗi epoch.

        Args:
            epochs: tổng số epochs (thường 30)

        Returns:
            list[dict] — history metrics cho từng epoch
        """
        logger.info("=" * 60)
        logger.info("TRAINING START")
        logger.info("=" * 60)
        logger.info(f"Epochs: {self.start_epoch + 1} → {epochs}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples:   {len(self.val_loader.dataset)}")
        logger.info(f"Device: {self.device}")
        logger.info(f"AMP: {self.use_amp}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")
        logger.info("=" * 60)

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            # ─── Train ───
            train_metrics = self.train_one_epoch(epoch)

            # ─── Validate ───
            val_metrics = self.validate(epoch)

            # ─── Scheduler step ───
            self.scheduler.step()

            # ─── Combine metrics ───
            lr = self.optimizer.param_groups[0]["lr"]
            metrics = {
                **train_metrics,
                **val_metrics,
                "lr": lr,
                "epoch": epoch,
                "epoch_time": time.time() - epoch_start,
            }
            self.history.append(metrics)

            # ─── Log to console ───
            self._log_epoch(metrics, epoch)

            # ─── Log to W&B ───
            if self.use_wandb:
                self._log_wandb(metrics, epoch)

            # ─── Early stopping ───
            monitor_key = "val_auc"
            monitor_value = val_metrics.get(monitor_key, 0.0)
            self.early_stopping(monitor_value)

            # ─── Save checkpoints ───
            if self.early_stopping.is_best:
                self.best_metric = monitor_value
                self.save_checkpoint(epoch, is_best=True)

            # Always save last (cho resume)
            self.save_checkpoint(epoch, is_best=False)

            if self.early_stopping.should_stop:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {self.early_stopping.patience} epochs)"
                )
                break

        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE — Best val_auc: {self.best_metric:.4f}")
        logger.info("=" * 60)

        return self.history

    # ──────────────────────────────────────────────────────────
    # LOGGING
    # ──────────────────────────────────────────────────────────

    def _log_epoch(self, metrics: dict, epoch: int) -> None:
        """Print metrics đẹp ra console."""
        best_marker = " ★" if self.early_stopping.is_best else ""
        logger.info(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Val Acc: {metrics['val_acc']:.4f} | "
            f"Val AUC: {metrics['val_auc']:.4f}{best_marker} | "
            f"LR: {metrics['lr']:.2e} | "
            f"{metrics['epoch_time']:.1f}s"
        )

    def _log_wandb(self, metrics: dict, epoch: int) -> None:
        """Log metrics lên W&B dashboard."""
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=epoch)
        except ImportError:
            pass

    # ──────────────────────────────────────────────────────────
    # CHECKPOINT SAVE / LOAD
    # ──────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Lưu checkpoint (model + optimizer + scheduler + state).

        Args:
            epoch: epoch hiện tại
            is_best: True → save thêm best.pt
        """
        # Unwrap DataParallel if needed — save clean state_dict without 'module.' prefix
        raw_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        state = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "early_stopping_state_dict": self.early_stopping.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
            "scaler_state_dict": (
                self.scaler.state_dict() if self.use_amp else None
            ),
        }

        # Always save last (cho resume)
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(state, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            logger.info(
                f"  ★ New best model saved (val_auc={self.best_metric:.4f})"
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint và resume training.

        Khôi phục TOÀN BỘ state: model, optimizer, scheduler, early_stopping.
        Training sẽ tiếp tục từ epoch tiếp theo.

        Args:
            checkpoint_path: đường dẫn tới file .pt
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "early_stopping_state_dict" in checkpoint:
            self.early_stopping.load_state_dict(
                checkpoint["early_stopping_state_dict"]
            )

        self.best_metric = checkpoint.get("best_metric", 0.0)
        self.start_epoch = checkpoint["epoch"] + 1

        if self.use_amp and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Resumed from epoch {self.start_epoch}, "
            f"best_metric={self.best_metric:.4f}"
        )
