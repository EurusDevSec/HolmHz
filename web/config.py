"""Web Demo configuration."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths
ONNX_MODEL_PATH = str(PROJECT_ROOT / "outputs" / "exports" / "efficientnet_b0.onnx")
PYTORCH_CHECKPOINT = str(PROJECT_ROOT / "outputs" / "checkpoints" / "best_v4.pt")
MODEL_NAME = "efficientnet_b0"

# Inference settings
THRESHOLD = 0.76  # Youden's J optimal from v4
DEVICE = "cpu"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
