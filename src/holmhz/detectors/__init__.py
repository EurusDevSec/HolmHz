# src/holmhz/detectors/__init__.py
"""Detector modules — Backbone + Head models for deepfake detection."""

from functools import partial

from ..utils.registry import DETECTOR_REGISTRY
from .base import BaseDetector
from .efficientnet_detector import EfficientNetDetector
from .timm_detector import TimmDetector

# Đăng ký detector vào registry
# Sau này thêm CLIP: import CLIPDetector + register
DETECTOR_REGISTRY.register("efficientnet_b0")(EfficientNetDetector)
DETECTOR_REGISTRY.register("resnet18")(partial(TimmDetector, model_name="resnet18"))
DETECTOR_REGISTRY.register("vit_small")(partial(TimmDetector, model_name="vit_small_patch16_224"))
DETECTOR_REGISTRY.register("swin_tiny")(partial(TimmDetector, model_name="swin_tiny_patch4_window7_224"))

__all__ = ["BaseDetector", "EfficientNetDetector", "TimmDetector", "DETECTOR_REGISTRY"]
