# src/holmhz/detectors/__init__.py
"""Detector modules — Backbone + Head models for deepfake detection."""

from functools import partial

from ..utils.registry import DETECTOR_REGISTRY
from .base import BaseDetector
from .efficientnet_detector import EfficientNetDetector
from .timm_detector import TimmDetector

# Đăng ký detector vào registry
DETECTOR_REGISTRY.register("efficientnet_b0")(EfficientNetDetector)
DETECTOR_REGISTRY.register("resnet18")(partial(TimmDetector, model_name="resnet18"))
DETECTOR_REGISTRY.register("vit_small")(partial(TimmDetector, model_name="vit_small_patch16_224"))
DETECTOR_REGISTRY.register("swin_tiny")(partial(TimmDetector, model_name="swin_tiny_patch4_window7_224"))

# CLIP detector — optional (requires open-clip-torch)
try:
    from .clip_detector import CLIPDetector
    DETECTOR_REGISTRY.register("clip_vit_l14")(CLIPDetector)
except ImportError:
    pass  # open_clip not installed

# Frequency detector — FFT-based (no extra dependencies)
from .freq_detector import FrequencyDetector
DETECTOR_REGISTRY.register("freq_fft")(FrequencyDetector)

__all__ = ["BaseDetector", "EfficientNetDetector", "TimmDetector", "FrequencyDetector", "DETECTOR_REGISTRY"]

