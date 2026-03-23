"""
Generic Timm Backbone — Wraps any timm model as a BaseBackbone.

Hỗ trợ tất cả models trong timm (700+), tự động detect features_dim.
Dùng chung cho ResNet-18, ViT-Small/16, Swin-T, v.v.

Usage:
    backbone = TimmBackbone("resnet18", pretrained=True)
    features = backbone.extract_features(x)  # [B, 512]

    backbone = TimmBackbone("vit_small_patch16_224", pretrained=True)
    features = backbone.extract_features(x)  # [B, 384]

    backbone = TimmBackbone("swin_tiny_patch4_window7_224", pretrained=True)
    features = backbone.extract_features(x)  # [B, 768]
"""

import timm
import torch

from .base import BaseBackbone


class TimmBackbone(BaseBackbone):
    """Generic backbone wrapping any timm model.

    Args:
        model_name: timm model name (e.g. "resnet18", "vit_small_patch16_224")
        pretrained: Load pretrained ImageNet weights
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()

        self.model_name = model_name
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head → feature extractor only
        )

        # Auto-detect features dimension from timm model
        self._features_dim = self.model.num_features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector from images.

        Args:
            x: [B, 3, 224, 224] — batch of normalized images

        Returns:
            [B, features_dim] — feature vector per image
        """
        return self.model(x)

    def get_features_dim(self) -> int:
        """Return feature dimension of the backbone."""
        return self._features_dim
