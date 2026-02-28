"""Tests cho Evaluator class."""

import pytest
import torch
from torch.utils.data import DataLoader

from holmhz.evaluation import Evaluator


class FakeModel(torch.nn.Module):
    """Model giả — trả logits cố định cho testing."""

    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        self._predictions = predictions
        self._idx = 0

    def forward(self, x):
        batch_size = x.shape[0]
        logits = self._predictions[self._idx : self._idx + batch_size]
        self._idx += batch_size
        return logits.unsqueeze(-1)


class FakeDataset(torch.utils.data.Dataset):
    """Dataset giả — trả dict giống ImageDataset."""

    def __init__(self, images, labels, sources):
        self.images = images
        self.labels = labels
        self.sources = sources

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "source": self.sources[idx],
            "path": f"test_{idx}.jpg",
        }


class TestEvaluator:
    """Test Evaluator class."""

    def _make_evaluator(self, logits, labels, sources):
        """Helper: tạo Evaluator với data giả."""
        images = torch.randn(len(labels), 3, 224, 224)
        dataset = FakeDataset(images, labels, sources)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = FakeModel(logits)
        return Evaluator(model, loader, torch.device("cpu"))

    def test_perfect_predictions(self):
        """Model dự đoán hoàn hảo → AUC = 1.0."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        sources = ["src_a", "src_a", "src_b", "src_b"]

        evaluator = self._make_evaluator(logits, labels, sources)
        results = evaluator.evaluate()

        assert results["overall"]["auc"] == pytest.approx(1.0, abs=0.01)
        assert results["overall"]["accuracy"] == pytest.approx(1.0, abs=0.01)
        assert results["total"] == 4

    def test_per_source_breakdown(self):
        """Per-source metrics được tính riêng."""
        logits = torch.tensor([3.0, -3.0, 3.0, -3.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        sources = ["cifake", "cifake", "ffhq", "ffhq"]

        evaluator = self._make_evaluator(logits, labels, sources)
        results = evaluator.evaluate()

        assert "cifake" in results["per_source"]
        assert "ffhq" in results["per_source"]
        assert results["per_source"]["cifake"]["n"] == 2
        assert results["per_source"]["ffhq"]["n"] == 2

    def test_returns_raw_data(self):
        """Evaluator trả về raw logits/labels/sources cho visualization."""
        logits = torch.tensor([1.0, -1.0])
        labels = torch.tensor([1.0, 0.0])
        sources = ["src_a", "src_b"]

        evaluator = self._make_evaluator(logits, labels, sources)
        results = evaluator.evaluate()

        assert "all_logits" in results
        assert "all_labels" in results
        assert "all_sources" in results
        assert len(results["all_logits"]) == 2

    def test_single_class_source(self):
        """Source chỉ có 1 class → AUC = 0.5 (edge case)."""
        logits = torch.tensor([3.0, 3.0, -3.0, -3.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        sources = ["fake_only", "fake_only", "real_only", "real_only"]

        evaluator = self._make_evaluator(logits, labels, sources)
        results = evaluator.evaluate()

        # fake_only source: chỉ label=1 → AUC = 0.5
        assert results["per_source"]["fake_only"]["auc"] == 0.5
        assert results["per_source"]["real_only"]["auc"] == 0.5
