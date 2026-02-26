# src/holmhz/backbones/__init__.py
"""Backbone modules — CNN feature extractors."""

from ..utils.registry import BACKBONE_REGISTRY
from .base import BaseBackbone
from .efficientnet import EfficientNetBackbone

# Đăng ký backbone vào registry
BACKBONE_REGISTRY.register("efficientnet_b0")(EfficientNetBackbone)

__all__ = ["BaseBackbone", "EfficientNetBackbone", "BACKBONE_REGISTRY"]
