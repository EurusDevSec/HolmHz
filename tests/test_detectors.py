"""Unit tests cho detector modules.

Kiểm tra:
- EfficientNetDetector tạo được, forward pass đúng shape
- Freeze backbone: chỉ head trainable
- Unfreeze backbone: toàn bộ trainable
- predict_proba trả về [0, 1]
- predict trả về 0 hoặc 1
- Registry hoạt động
- Tích hợp với DataLoader (batch shape)
"""

import pytest
import torch

from holmhz.detectors import EfficientNetDetector
from holmhz.detectors.base import BaseDetector


class TestBaseDetector:
    """Test abstract base class."""

    def test_cannot_instantiate_base(self):
        """BaseDetector là abstract — không thể tạo instance trực tiếp."""
        with pytest.raises(TypeError):
            BaseDetector()

    def test_efficientnet_is_subclass(self):
        """EfficientNetDetector kế thừa BaseDetector."""
        assert issubclass(EfficientNetDetector, BaseDetector)


class TestEfficientNetDetector:
    """Test EfficientNet-B0 detector."""

    @pytest.fixture
    def model_frozen(self):
        """Detector với backbone frozen (Phase 1)."""
        return EfficientNetDetector(
            pretrained=False,
            dropout=0.3,
            freeze_backbone=True,
        )

    @pytest.fixture
    def model_unfrozen(self):
        """Detector với backbone unfrozen (Phase 2)."""
        return EfficientNetDetector(
            pretrained=False,
            dropout=0.3,
            freeze_backbone=False,
        )

    @pytest.fixture
    def dummy_input(self):
        """Batch 4 ảnh giả 224×224."""
        return torch.randn(4, 3, 224, 224)

    # --- Forward pass ---

    def test_forward_shape(self, model_frozen, dummy_input):
        """Forward phải trả về [B, 1]."""
        model_frozen.eval()
        with torch.no_grad():
            logits = model_frozen(dummy_input)
        assert logits.shape == (4, 1)

    def test_forward_dtype(self, model_frozen, dummy_input):
        """Output phải là float32."""
        model_frozen.eval()
        with torch.no_grad():
            logits = model_frozen(dummy_input)
        assert logits.dtype == torch.float32

    def test_forward_returns_logits(self, model_frozen, dummy_input):
        """Forward trả về logits (có thể âm hoặc dương, không ép [0,1])."""
        model_frozen.eval()
        with torch.no_grad():
            logits = model_frozen(dummy_input)
        # Logits là raw scores — có thể nằm ngoài [0, 1]
        # Không kiểm tra range cụ thể vì random input
        assert logits.shape == (4, 1)

    def test_single_image(self, model_frozen):
        """Phải hoạt động với batch_size=1."""
        model_frozen.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model_frozen(x)
        assert logits.shape == (1, 1)

    # --- Predict methods ---

    def test_predict_proba_range(self, model_frozen, dummy_input):
        """predict_proba phải trả về giá trị trong [0, 1]."""
        probs = model_frozen.predict_proba(dummy_input)
        assert probs.shape == (4, 1)
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

    def test_predict_labels(self, model_frozen, dummy_input):
        """predict phải trả về 0 hoặc 1."""
        labels = model_frozen.predict(dummy_input)
        assert labels.shape == (4, 1)
        assert set(labels.flatten().tolist()).issubset({0, 1})

    # --- Freeze / Unfreeze ---

    def test_frozen_backbone_trainable_params(self, model_frozen):
        """Khi freeze backbone, chỉ head params là trainable."""
        trainable = sum(
            p.numel() for p in model_frozen.parameters() if p.requires_grad
        )
        # Head = Linear(1280, 1) + bias = 1280 + 1 = 1,281
        assert trainable == 1281, f"Expected 1281 trainable params, got {trainable}"

    def test_unfrozen_all_trainable(self, model_unfrozen):
        """Khi unfreeze, tất cả params là trainable."""
        total = sum(p.numel() for p in model_unfrozen.parameters())
        trainable = sum(
            p.numel() for p in model_unfrozen.parameters() if p.requires_grad
        )
        assert total == trainable

    def test_total_params(self, model_frozen):
        """Total params phải ~4M (backbone 4M + head 1.3K)."""
        total = sum(p.numel() for p in model_frozen.parameters())
        # EfficientNet-B0 + head: 4,008,829
        assert 3_500_000 < total < 5_000_000, f"Unexpected total: {total:,}"
        assert total <= 6_000_000, "Exceeds 6M param limit from AC"

    def test_unfreeze_backbone(self, model_frozen):
        """Có thể unfreeze backbone sau khi tạo model."""
        # Kiểm tra frozen
        trainable_before = sum(
            p.numel() for p in model_frozen.parameters() if p.requires_grad
        )
        assert trainable_before == 1281

        # Unfreeze
        model_frozen.backbone.unfreeze()
        trainable_after = sum(
            p.numel() for p in model_frozen.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in model_frozen.parameters())
        assert trainable_after == total

    # --- Grad-CAM layer ---

    def test_get_feature_layer(self, model_frozen):
        """get_feature_layer phải trả về nn.Module (cho Grad-CAM)."""
        layer = model_frozen.get_feature_layer()
        assert isinstance(layer, torch.nn.Module)

    # --- Gradient flow ---

    def test_gradient_flows_through_head(self, model_frozen, dummy_input):
        """Gradient phải chạy qua head khi backbone frozen."""
        logits = model_frozen(dummy_input)
        loss = logits.sum()
        loss.backward()

        # Head params phải có gradient
        for name, param in model_frozen.head.named_parameters():
            assert param.grad is not None, f"No gradient for head param: {name}"

    def test_no_gradient_frozen_backbone(self, model_frozen, dummy_input):
        """Backbone frozen → không có gradient cho backbone params."""
        logits = model_frozen(dummy_input)
        loss = logits.sum()
        loss.backward()

        # Backbone params KHÔNG có gradient
        for param in model_frozen.backbone.parameters():
            assert param.grad is None


class TestDetectorRegistry:
    """Test Registry pattern cho detectors."""

    def test_registry_build(self):
        """Phải tạo được model qua registry."""
        # Ensure registration happened
        import holmhz.detectors  # noqa: F401
        from holmhz.utils.registry import DETECTOR_REGISTRY

        model = DETECTOR_REGISTRY.build(
            "efficientnet_b0", pretrained=False, freeze_backbone=True
        )
        assert isinstance(model, EfficientNetDetector)

    def test_registry_list(self):
        """Registry phải list được các detectors đã đăng ký."""
        import holmhz.detectors  # noqa: F401
        from holmhz.utils.registry import DETECTOR_REGISTRY

        detectors = DETECTOR_REGISTRY.list()
        assert "efficientnet_b0" in detectors

    def test_registry_unknown_raises(self):
        """Tên không tồn tại phải raise KeyError."""
        from holmhz.utils.registry import DETECTOR_REGISTRY

        with pytest.raises(KeyError, match="not_a_real_model"):
            DETECTOR_REGISTRY.build("not_a_real_model")

    def test_registry_contains(self):
        """Registry hỗ trợ 'in' operator."""
        import holmhz.detectors  # noqa: F401
        from holmhz.utils.registry import DETECTOR_REGISTRY

        assert "efficientnet_b0" in DETECTOR_REGISTRY
