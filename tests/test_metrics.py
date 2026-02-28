"""Tests cho metrics module — F1, Precision, Recall."""

import pytest
import torch

from holmhz.metrics import compute_accuracy, compute_auc
from holmhz.metrics.f1 import compute_f1
from holmhz.metrics.precision import compute_precision
from holmhz.metrics.recall import compute_recall


class TestPrecision:
    """Test compute_precision."""

    def test_perfect_precision(self):
        """Tất cả dự đoán FAKE đều đúng → precision = 1.0."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_precision(logits, labels) == pytest.approx(1.0, abs=0.01)

    def test_low_precision(self):
        """Nhiều False Positive → precision thấp."""
        # 2 TP + 2 FP → precision = 2/4 = 0.5
        logits = torch.tensor([3.0, 3.0, 3.0, 3.0])  # All predict FAKE
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])   # But 2 are Real
        assert compute_precision(logits, labels) == pytest.approx(0.5, abs=0.01)

    def test_no_positive_predictions(self):
        """Model không dự đoán FAKE nào → precision = 0.0."""
        logits = torch.tensor([-3.0, -3.0, -3.0])
        labels = torch.tensor([1.0, 1.0, 0.0])
        assert compute_precision(logits, labels) == 0.0

    def test_2d_logits(self):
        """Logits shape [N, 1] cũng hoạt động."""
        logits = torch.tensor([[3.0], [-3.0]])
        labels = torch.tensor([1.0, 0.0])
        assert compute_precision(logits, labels) == pytest.approx(1.0, abs=0.01)


class TestRecall:
    """Test compute_recall."""

    def test_perfect_recall(self):
        """Tìm ra tất cả Fake → recall = 1.0."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_recall(logits, labels) == pytest.approx(1.0, abs=0.01)

    def test_low_recall(self):
        """Bỏ sót nhiều Fake → recall thấp."""
        # 1 TP + 1 FN → recall = 1/2 = 0.5
        logits = torch.tensor([3.0, -3.0])
        labels = torch.tensor([1.0, 1.0])  # Both Fake, but only 1 detected
        assert compute_recall(logits, labels) == pytest.approx(0.5, abs=0.01)

    def test_no_actual_fakes(self):
        """Không có Fake nào trong data → recall = 0.0."""
        logits = torch.tensor([3.0, -3.0])
        labels = torch.tensor([0.0, 0.0])  # All Real
        assert compute_recall(logits, labels) == 0.0


class TestF1:
    """Test compute_f1."""

    def test_perfect_f1(self):
        """Precision và Recall đều = 1.0 → F1 = 1.0."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_f1(logits, labels) == pytest.approx(1.0, abs=0.01)

    def test_zero_f1(self):
        """Prediction hoàn toàn sai → F1 = 0.0."""
        logits = torch.tensor([-3.0, -3.0])  # All predict Real
        labels = torch.tensor([1.0, 1.0])     # All actually Fake
        assert compute_f1(logits, labels) == 0.0

    def test_balanced_f1(self):
        """Precision=Recall=0.5 → F1=0.5."""
        # 1 TP, 1 FP, 1 FN → Prec=0.5, Rec=0.5, F1=0.5
        logits = torch.tensor([3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0])
        assert compute_f1(logits, labels) == pytest.approx(0.5, abs=0.05)

    def test_consistency_with_precision_recall(self):
        """F1 = 2 * (P * R) / (P + R)."""
        logits = torch.tensor([3.0, 3.0, -3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        p = compute_precision(logits, labels)
        r = compute_recall(logits, labels)
        expected_f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        assert compute_f1(logits, labels) == pytest.approx(expected_f1, abs=0.01)
