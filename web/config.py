"""Web Demo configuration — ResNet-18 v2 (best overall: ID AUC 0.9953, OOD AUC 0.8646)."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths — ResNet-18 v2 (trained on raw_v2, passes ALL 4 KPIs)
ONNX_MODEL_PATH = str(PROJECT_ROOT / "outputs" / "exports" / "resnet18_v2.onnx")
PYTORCH_CHECKPOINT = str(PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet18_v2_clean.pt")
MODEL_NAME = "resnet18"

# CLIP v9 (optional ensemble, disabled by default for ResNet-18 demo)
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
