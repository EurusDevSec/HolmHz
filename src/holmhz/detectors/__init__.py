# src/holmhz/detectors/__init__.py
"""Detector modules — Backbone + Head models for deepfake detection."""

from ..utils.registry import DETECTOR_REGISTRY
from .base import BaseDetector
from .efficientnet_detector import EfficientNetDetector

# Đăng ký detector vào registry
# Sau này thêm CLIP: import CLIPDetector + register
DETECTOR_REGISTRY.register("efficientnet_b0")(EfficientNetDetector)

__all__ = ["BaseDetector", "EfficientNetDetector", "DETECTOR_REGISTRY"]
