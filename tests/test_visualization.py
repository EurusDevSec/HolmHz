"""Tests cho visualization module."""

import os

import pytest
import torch

from holmhz.utils.visualization import (
    plot_confusion_matrix,
    plot_per_source_accuracy,
    plot_roc_curve,
)


@pytest.fixture
def tmp_dir(tmp_path):
    """Thư mục tạm cho output."""
    return str(tmp_path)


class TestConfusionMatrix:
    """Test plot_confusion_matrix."""

    def test_saves_file(self, tmp_dir):
        """Tạo file PNG thành công."""
        labels = torch.tensor([0, 0, 1, 1])
        logits = torch.tensor([-3.0, -3.0, 3.0, 3.0])
        path = os.path.join(tmp_dir, "cm.png")

        result = plot_confusion_matrix(labels, logits, path)

        assert os.path.exists(result)
        assert result.endswith(".png")

    def test_creates_directory(self, tmp_dir):
        """Tự tạo thư mục nếu chưa có."""
        labels = torch.tensor([0, 1])
        logits = torch.tensor([-3.0, 3.0])
        path = os.path.join(tmp_dir, "subdir", "cm.png")

        result = plot_confusion_matrix(labels, logits, path)
        assert os.path.exists(result)


class TestROCCurve:
    """Test plot_roc_curve."""

    def test_saves_file(self, tmp_dir):
        """Tạo file PNG thành công."""
        results = {
            "Test": {
                "all_logits": torch.tensor([3.0, -3.0, 2.0, -2.0]),
                "all_labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            }
        }
        path = os.path.join(tmp_dir, "roc.png")

        result = plot_roc_curve(results, path)
        assert os.path.exists(result)

    def test_multiple_curves(self, tmp_dir):
        """Vẽ nhiều curves chồng lên."""
        results = {
            "ID": {
                "all_logits": torch.tensor([3.0, -3.0, 2.0, -2.0]),
                "all_labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            },
            "OOD": {
                "all_logits": torch.tensor([1.0, 1.0, -1.0, -1.0]),
                "all_labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            },
        }
        path = os.path.join(tmp_dir, "roc_multi.png")

        result = plot_roc_curve(results, path)
        assert os.path.exists(result)


class TestPerSourceAccuracy:
    """Test plot_per_source_accuracy."""

    def test_saves_file(self, tmp_dir):
        """Tạo file PNG thành công."""
        per_source = {
            "cifake": {"accuracy": 0.99, "n": 2100},
            "real_camera": {"accuracy": 0.05, "n": 100},
        }
        path = os.path.join(tmp_dir, "per_source.png")

        result = plot_per_source_accuracy(per_source, path)
        assert os.path.exists(result)
