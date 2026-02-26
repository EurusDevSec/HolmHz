"""
EfficientNet-B0 Deepfake Detector — Model chính của HolmHz.

Kiến trúc tổng thể:
    Input [B, 3, 224, 224]
    → EfficientNet-B0 Backbone [B, 1280]  (Pretrained ImageNet, 4M params)
    → Dropout(0.3)                         (Chống overfitting)
    → Linear(1280, 1)                      (1 output = raw logit)
    → Output [B, 1]                        (Logits — chưa sigmoid)

Training: loss = BCEWithLogitsLoss(logits, labels)
Inference: probs = sigmoid(logits) → P(Fake)

Tại sao KHÔNG có Sigmoid trong forward()?
→ BCEWithLogitsLoss tự tính Sigmoid bên trong
→ Numerical stability tốt hơn (tránh log(0) và saturated gradients)
→ Pattern chuẩn trong PyTorch deep learning

Params breakdown:
    Backbone: 4,007,548
    Head:     1,281 (Linear: 1280*1 + 1 bias)
    Total:    4,008,829 (~4M, well under 6M limit)
"""

import torch
import torch.nn as nn

from ..backbones.efficientnet import EfficientNetBackbone
from .base import BaseDetector


class EfficientNetDetector(BaseDetector):
    """Detector sử dụng EfficientNet-B0 backbone.

    Args:
        pretrained: Load pretrained ImageNet weights cho backbone
        dropout: Tỷ lệ dropout (0.3 = tắt 30% neuron ngẫu nhiên khi train)
        freeze_backbone: Đóng băng backbone (Phase 1 transfer learning)

    Example:
        >>> model = EfficientNetDetector(pretrained=True, freeze_backbone=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> logits = model(x)  # [4, 1]
        >>> probs = model.predict_proba(x)  # [4, 1] — P(Fake) ∈ [0, 1]
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Backbone: EfficientNet-B0 feature extractor
        self.backbone = EfficientNetBackbone(pretrained=pretrained)

        # Head: Classification layers
        # Dropout → Linear, KHÔNG có Sigmoid (BCEWithLogitsLoss xử lý)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.get_features_dim(), 1),
        )

        # Freeze backbone nếu được yêu cầu (Phase 1)
        if freeze_backbone:
            self.backbone.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ảnh → logits.

        Args:
            x: [B, 3, 224, 224] — batch ảnh đã normalize (ImageNet stats)

        Returns:
            logits: [B, 1] — raw scores

        QUAN TRỌNG: Output là LOGITS (chưa sigmoid).
        - Training: BCEWithLogitsLoss(logits, labels) — loss tự sigmoid
        - Inference: dùng predict_proba(x) để lấy P(Fake) ∈ [0,1]
        """
        features = self.backbone.extract_features(x)  # [B, 1280]
        logits = self.head(features)                    # [B, 1]
        return logits

    def get_feature_layer(self) -> nn.Module:
        """Trả về layer cuối của backbone — dùng cho Grad-CAM (Task 2.3).

        Grad-CAM cần "nhìn vào" layer convolution cuối cùng
        để tạo heatmap giải thích model đang nhìn vùng nào.

        Returns:
            nn.Module: conv_head layer của EfficientNet-B0
        """
        return self.backbone.model.conv_head
