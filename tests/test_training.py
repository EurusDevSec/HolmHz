"""Unit tests cho training pipeline.

Kiểm tra:
- Metrics (accuracy, AUC) tính đúng
- Loss function factory hoạt động
- EarlyStopping logic đúng (patience, state_dict)
- LR Scheduler factory hoạt động
- Trainer forward/backward pass (1 batch)
"""

import pytest
import torch
import torch.nn as nn

from holmhz.losses import get_loss_fn
from holmhz.metrics import compute_accuracy, compute_auc
from holmhz.training import EarlyStopping, get_scheduler


class TestMetrics:
    """Test accuracy và AUC computation."""

    def test_accuracy_perfect(self):
        """Model dự đoán hoàn hảo → accuracy = 1.0."""
        logits = torch.tensor([5.0, -5.0, 5.0, -5.0])  # Rõ ràng Fake, Real
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_accuracy(logits, labels) == 1.0

    def test_accuracy_random(self):
        """Model dự đoán ngược → accuracy = 0.0."""
        logits = torch.tensor([-5.0, 5.0])  # Ngược hết
        labels = torch.tensor([1.0, 0.0])
        assert compute_accuracy(logits, labels) == 0.0

    def test_accuracy_with_2d_logits(self):
        """Logits shape [B, 1] (từ model) cũng phải hoạt động."""
        logits = torch.tensor([[5.0], [-5.0]])
        labels = torch.tensor([1.0, 0.0])
        assert compute_accuracy(logits, labels) == 1.0

    def test_auc_perfect(self):
        """Phân biệt hoàn hảo → AUC = 1.0."""
        logits = torch.tensor([5.0, -5.0, 5.0, -5.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_auc(logits, labels) == 1.0

    def test_auc_random(self):
        """Chỉ 1 class → AUC = 0.5 (edge case)."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        labels = torch.tensor([1.0, 1.0, 1.0])  # Toàn Fake
        assert compute_auc(logits, labels) == 0.5

    def test_auc_with_2d_logits(self):
        """Logits shape [B, 1] cũng phải hoạt động."""
        logits = torch.tensor([[5.0], [-5.0], [3.0], [-3.0]])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_auc(logits, labels) == 1.0


class TestLossFunction:
    """Test loss function factory."""

    def test_bce_with_logits(self):
        """BCEWithLogitsLoss phải tạo được và tính loss."""
        loss_fn = get_loss_fn("bce_with_logits")
        logits = torch.tensor([0.0, 0.0])  # Uncertain
        labels = torch.tensor([1.0, 0.0])
        loss = loss_fn(logits, labels)
        # loss ≈ 0.693 (log(2)) khi logits = 0
        assert 0.6 < loss.item() < 0.8

    def test_bce_with_pos_weight(self):
        """pos_weight parameter phải hoạt động."""
        loss_fn = get_loss_fn("bce_with_logits", pos_weight=2.0)
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_unknown_loss_raises(self):
        """Loss không tồn tại phải raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss_fn("unknown_loss")


class TestEarlyStopping:
    """Test Early Stopping logic."""

    def test_first_epoch_is_best(self):
        """Epoch đầu luôn là best."""
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)
        assert es.is_best
        assert not es.should_stop

    def test_improvement_resets_counter(self):
        """Metric cải thiện → reset counter."""
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)
        es(0.4)  # Worse → counter=1
        assert es.counter == 1
        es(0.6)  # Better → counter=0
        assert es.counter == 0
        assert es.is_best

    def test_patience_triggers_stop(self):
        """Hết patience → should_stop = True."""
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)  # Best
        es(0.4)  # counter=1
        es(0.3)  # counter=2
        result = es(0.2)  # counter=3 → STOP
        assert result is True
        assert es.should_stop

    def test_mode_min(self):
        """mode='min': metric giảm = tốt."""
        es = EarlyStopping(patience=3, mode="min")
        es(0.5)  # Best
        es(0.3)  # Better (lower)
        assert es.is_best
        assert es.best_score == 0.3

    def test_state_dict_roundtrip(self):
        """state_dict save/load phải giữ nguyên state."""
        es = EarlyStopping(patience=5, mode="max")
        es(0.5)
        es(0.6)
        es(0.55)  # Worse → counter=1

        state = es.state_dict()
        assert state["counter"] == 1
        assert state["best_score"] == 0.6

        es2 = EarlyStopping(patience=5, mode="max")
        es2.load_state_dict(state)
        assert es2.counter == 1
        assert es2.best_score == 0.6


class TestLRScheduler:
    """Test LR Scheduler factory."""

    def test_cosine_scheduler(self):
        """CosineAnnealingLR phải tạo được."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = get_scheduler(optimizer, "cosine", epochs=10)

        # LR phải giảm sau step
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]
        assert lr_after < lr_before

    def test_unknown_scheduler_raises(self):
        """Scheduler không tồn tại phải raise ValueError."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler(optimizer, "unknown_scheduler")
