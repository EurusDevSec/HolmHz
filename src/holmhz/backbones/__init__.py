# src/holmhz/backbones/__init__.py
"""Backbone modules — CNN feature extractors."""

from functools import partial

from ..utils.registry import BACKBONE_REGISTRY
from .base import BaseBackbone
from .efficientnet import EfficientNetBackbone
from .timm_backbone import TimmBackbone

# Đăng ký backbone vào registry
BACKBONE_REGISTRY.register("efficientnet_b0")(EfficientNetBackbone)
BACKBONE_REGISTRY.register("resnet18")(partial(TimmBackbone, model_name="resnet18"))
BACKBONE_REGISTRY.register("vit_small")(partial(TimmBackbone, model_name="vit_small_patch16_224"))
BACKBONE_REGISTRY.register("swin_tiny")(partial(TimmBackbone, model_name="swin_tiny_patch4_window7_224"))

__all__ = ["BaseBackbone", "EfficientNetBackbone", "TimmBackbone", "BACKBONE_REGISTRY"]
