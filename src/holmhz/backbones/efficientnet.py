"""
EfficientNet-B0 Backbone sử dụng thư viện timm.

timm (PyTorch Image Models): thư viện chứa 700+ pre-trained models.
Thay vì tự code kiến trúc EfficientNet (rất phức tạp), ta import từ timm.

Verified với timm v1.0.24:
- Output shape: [B, 1280] khi num_classes=0
- Backbone params: 4,007,548
- Bao gồm Global Average Pooling (không cần thêm)
"""

import timm
import torch

from .base import BaseBackbone


class EfficientNetBackbone(BaseBackbone):
    """EfficientNet-B0 feature extractor.

    Kiến trúc:
        Input [B, 3, 224, 224]
        → EfficientNet MBConv Layers (pretrained ImageNet)
        → Global Average Pooling
        → Output [B, 1280]  ← vector features

    Args:
        pretrained: Load pretrained ImageNet weights (default: True)

    Params: 4,007,548 (backbone only, không tính head)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load EfficientNet-B0 từ timm
        # num_classes=0 → bỏ lớp classification cuối
        # → chỉ lấy phần feature extractor + global pool
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # Bỏ FC cuối — ta tự thêm head ở Detector
        )

        self._features_dim = 1280  # EfficientNet-B0 output dimension

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất 1280-dim feature vector từ ảnh.

        Args:
            x: [B, 3, 224, 224] — batch ảnh đã normalize (ImageNet stats)

        Returns:
            [B, 1280] — feature vector cho mỗi ảnh

        Note:
            timm model đã bao gồm Global Average Pooling.
            Output đã được flatten thành 1D vector.
        """
        return self.model(x)

    def get_features_dim(self) -> int:
        """Trả về 1280 — feature dimension của EfficientNet-B0."""
        return self._features_dim
