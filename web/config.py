"""Web Demo configuration."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths — v6 (data reset, trained on rvf10k + ciplab)
ONNX_MODEL_PATH = str(PROJECT_ROOT / "outputs" / "exports" / "efficientnet_b0_v6.onnx")
PYTORCH_CHECKPOINT = str(PROJECT_ROOT / "outputs" / "checkpoints" / "best_v6.pt")
MODEL_NAME = "efficientnet_b0"

# Inference settings
THRESHOLD = 0.5  # Reset to 0.5 for v6 (was 0.76 for v4, biased)
DEVICE = "cpu"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
