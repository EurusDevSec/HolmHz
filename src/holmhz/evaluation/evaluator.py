"""
Evaluator — đánh giá model trên test set.

Evaluator chạy inference trên toàn bộ dataloader, thu thập
predictions, và tính metrics (overall + per-source breakdown).

Usage:
    evaluator = Evaluator(model, dataloader, device)
    results = evaluator.evaluate()
    print(results["overall"]["auc"])        # 0.9983
    print(results["per_source"]["cifake"])  # {"accuracy": 0.99, ...}
"""

from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from holmhz.metrics import (
    compute_accuracy,
    compute_auc,
)
from holmhz.metrics.f1 import compute_f1
from holmhz.metrics.precision import compute_precision
from holmhz.metrics.recall import compute_recall
from holmhz.utils.logger import get_logger

logger = get_logger("evaluator")


class Evaluator:
    """Đánh giá model trên 1 test set.

    Attributes:
        model: Trained model (nn.Module).
        dataloader: Test DataLoader (ImageDataset format).
        device: torch.device.
        threshold: Classification threshold (default 0.5).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        threshold: float = 0.5,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Chạy evaluation trên toàn bộ dataloader.

        Returns:
            dict chứa:
                - "overall": dict metrics tổng (auc, accuracy, f1, precision, recall)
                - "per_source": dict[source_name] → dict metrics
                - "all_logits": tensor [N] — raw logits
                - "all_labels": tensor [N] — ground truth
                - "all_sources": list[str] — source tag cho mỗi sample
                - "total": int — tổng số samples
        """
        self.model.eval()
        self.model.to(self.device)

        all_logits = []
        all_labels = []
        all_sources = []

        logger.info(
            f"Evaluating {len(self.dataloader.dataset)} samples "
            f"({len(self.dataloader)} batches)..."
        )

        for batch in tqdm(self.dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"]
            sources = batch["source"]

            # Inference
            logits = self.model(images).squeeze(-1)  # [B, 1] → [B]

            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_sources.extend(sources)

        # Concatenate all batches
        all_logits = torch.cat(all_logits)  # [N]
        all_labels = torch.cat(all_labels)  # [N]

        # ─── Overall metrics ───
        overall = self._compute_metrics(all_logits, all_labels)
        logger.info(
            f"Overall — AUC: {overall['auc']:.4f}, "
            f"Acc: {overall['accuracy']:.4f}, "
            f"F1: {overall['f1']:.4f}"
        )

        # ─── Per-source metrics ───
        per_source = self._compute_per_source(all_logits, all_labels, all_sources)
        for source, metrics in per_source.items():
            logger.info(
                f"  {source:20s} — "
                f"Acc: {metrics['accuracy']:.4f}, "
                f"N: {metrics['n']}"
            )

        return {
            "overall": overall,
            "per_source": per_source,
            "all_logits": all_logits,
            "all_labels": all_labels,
            "all_sources": all_sources,
            "total": len(all_logits),
        }

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Tính tất cả 5 metrics cho 1 tập logits/labels."""
        return {
            "auc": compute_auc(logits, labels),
            "accuracy": compute_accuracy(logits, labels, self.threshold),
            "f1": compute_f1(logits, labels, self.threshold),
            "precision": compute_precision(logits, labels, self.threshold),
            "recall": compute_recall(logits, labels, self.threshold),
        }

    def _compute_per_source(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
        all_sources: list,
    ) -> dict:
        """Tính metrics riêng cho mỗi nguồn dữ liệu.

        Nhóm samples theo source tag, tính metrics cho mỗi nhóm.
        """
        # Nhóm indices theo source
        source_indices = defaultdict(list)
        for i, src in enumerate(all_sources):
            source_indices[src].append(i)

        per_source = {}
        for source, indices in sorted(source_indices.items()):
            idx = torch.tensor(indices)
            src_logits = all_logits[idx]
            src_labels = all_labels[idx]

            metrics = self._compute_metrics(src_logits, src_labels)
            metrics["n"] = len(indices)

            # Thêm label distribution info
            n_real = int((src_labels == 0).sum())
            n_fake = int((src_labels == 1).sum())
            metrics["n_real"] = n_real
            metrics["n_fake"] = n_fake

            per_source[source] = metrics

        return per_source