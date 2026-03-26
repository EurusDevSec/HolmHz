"""Web Demo configuration — Ensemble v8 EfficientNet + v9 CLIP."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths — Ensemble
# EfficientNet v8 (ID: AUC 0.9984, Acc 98.7%)
ONNX_MODEL_PATH = str(PROJECT_ROOT / "outputs" / "exports" / "efficientnet_b0_v8.onnx")
PYTORCH_CHECKPOINT = str(PROJECT_ROOT / "outputs" / "checkpoints" / "best_v7.pt")
MODEL_NAME = "efficientnet_b0"

# CLIP v9 (OOD: AUC 0.9419, camera_real 80.9%)
CLIP_CHECKPOINT = str(PROJECT_ROOT / "outputs" / "checkpoints" / "best_v9_clip.pt")

# Inference settings
THRESHOLD = 0.5
DEVICE = "cpu"

# Ensemble weights (CLIP higher → better OOD robustness)
EFFNET_WEIGHT = 0.4
CLIP_WEIGHT = 0.6

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
