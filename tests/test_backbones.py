"""Unit tests cho backbone modules.

Kiểm tra:
- EfficientNetBackbone tạo được, extract features đúng shape
- Freeze/unfreeze hoạt động đúng
- Features dimension đúng (1280)
- Forward pass = extract_features
"""

import pytest
import torch

from holmhz.backbones import EfficientNetBackbone
from holmhz.backbones.base import BaseBackbone


class TestBaseBackbone:
    """Test abstract base class."""

    def test_cannot_instantiate_base(self):
        """BaseBackbone là abstract — không thể tạo instance trực tiếp."""
        with pytest.raises(TypeError):
            BaseBackbone()

    def test_efficientnet_is_subclass(self):
        """EfficientNetBackbone kế thừa BaseBackbone."""
        assert issubclass(EfficientNetBackbone, BaseBackbone)


class TestEfficientNetBackbone:
    """Test EfficientNet-B0 backbone."""

    @pytest.fixture
    def backbone(self):
        """Tạo backbone không pretrained (nhanh hơn cho test)."""
        return EfficientNetBackbone(pretrained=False)

    @pytest.fixture
    def dummy_input(self):
        """Batch 4 ảnh giả 224×224."""
        return torch.randn(4, 3, 224, 224)

    def test_features_dim(self, backbone):
        """Feature dimension phải là 1280."""
        assert backbone.get_features_dim() == 1280

    def test_extract_features_shape(self, backbone, dummy_input):
        """extract_features phải trả về [B, 1280]."""
        features = backbone.extract_features(dummy_input)
        assert features.shape == (4, 1280)

    def test_forward_equals_extract_features(self, backbone, dummy_input):
        """forward() phải cho kết quả giống extract_features()."""
        backbone.eval()
        with torch.no_grad():
            f1 = backbone.forward(dummy_input)
            f2 = backbone.extract_features(dummy_input)
        assert torch.allclose(f1, f2)

    def test_forward_call_syntax(self, backbone, dummy_input):
        """Có thể gọi backbone(x) thay vì backbone.forward(x)."""
        backbone.eval()
        with torch.no_grad():
            features = backbone(dummy_input)
        assert features.shape == (4, 1280)

    def test_freeze(self, backbone):
        """Freeze phải tắt requires_grad cho tất cả params."""
        backbone.freeze()
        for param in backbone.parameters():
            assert not param.requires_grad

    def test_unfreeze(self, backbone):
        """Unfreeze phải bật requires_grad cho tất cả params."""
        backbone.freeze()  # Freeze trước
        backbone.unfreeze()  # Rồi unfreeze
        for param in backbone.parameters():
            assert param.requires_grad

    def test_param_count(self, backbone):
        """Backbone params phải ~ 4M (EfficientNet-B0)."""
        total = sum(p.numel() for p in backbone.parameters())
        # EfficientNet-B0 backbone: 4,007,548 params
        assert 3_500_000 < total < 5_000_000, f"Unexpected param count: {total:,}"

    def test_output_dtype(self, backbone, dummy_input):
        """Output phải là float32."""
        features = backbone.extract_features(dummy_input)
        assert features.dtype == torch.float32

    def test_single_image(self, backbone):
        """Phải hoạt động với batch_size=1."""
        x = torch.randn(1, 3, 224, 224)
        features = backbone.extract_features(x)
        assert features.shape == (1, 1280)
