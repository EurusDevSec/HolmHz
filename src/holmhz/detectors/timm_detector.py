"""
Generic Timm Detector — Backbone + Classification Head for deepfake detection.

Kiến trúc:
    Input [B, 3, 224, 224]
    → TimmBackbone (any timm model) [B, features_dim]
    → Dropout(p)
    → Linear(features_dim, 1)
    → Output [B, 1] (logits)

Hỗ trợ Grad-CAM qua get_feature_layer() — tự động chọn đúng layer
theo kiến trúc (CNN conv layer, ViT norm layer, Swin norm layer).
"""

import torch
import torch.nn as nn

from ..backbones.timm_backbone import TimmBackbone
from .base import BaseDetector

# Mapping từ model_name prefix → Grad-CAM target layer attribute path
# Mỗi kiến trúc có layer cuối khác nhau phù hợp cho Grad-CAM
_GRADCAM_LAYER_MAP = {
    "resnet": "layer4",           # ResNet: last residual block
    "vit_": "norm",               # ViT: final LayerNorm
    "swin_": "norm",              # Swin: final LayerNorm
    "efficientnet": "conv_head",  # EfficientNet: final conv
}


def _get_gradcam_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Resolve Grad-CAM target layer based on model architecture."""
    for prefix, attr in _GRADCAM_LAYER_MAP.items():
        if model_name.startswith(prefix):
            return getattr(model, attr)
    # Fallback: try common names
    for attr in ("norm", "layer4", "conv_head", "head"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError(f"Cannot determine Grad-CAM layer for {model_name}")


class TimmDetector(BaseDetector):
    """Generic detector using any timm backbone.

    Args:
        model_name: timm model name (e.g. "resnet18", "swin_tiny_patch4_window7_224")
        pretrained: Load pretrained ImageNet weights
        dropout: Dropout rate before classification head
        freeze_backbone: Freeze backbone weights (transfer learning Phase 1)
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.backbone = TimmBackbone(model_name=model_name, pretrained=pretrained)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.get_features_dim(), 1),
        )

        if freeze_backbone:
            self.backbone.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: images → logits.

        Args:
            x: [B, 3, 224, 224]

        Returns:
            [B, 1] — raw logits (no sigmoid)
        """
        features = self.backbone.extract_features(x)
        logits = self.head(features)
        return logits

    def get_feature_layer(self) -> nn.Module:
        """Return target layer for Grad-CAM visualization."""
        return _get_gradcam_layer(self.backbone.model, self.model_name)
