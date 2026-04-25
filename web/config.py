"""Web Demo configuration — EfficientNet-B0 v9 (BEST: ID AUC 0.9984, OOD AUC 0.8963).

Benchmark results (HolmHz-v2, Test ID: 3526, Test OOD: 182):
  EfficientNet-B0 v9 → ID AUC 0.9984 | OOD AUC 0.8963  ← THIS CONFIG
  ResNet-18 v2       → ID AUC 0.9953 | OOD AUC 0.8646
  ViT-Small/16 v2    → ID AUC 0.9741 | OOD AUC 0.8331
  Swin-Tiny v2†      → training failed
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths — EfficientNet-B0 v9 (best overall model, 4M params)
ONNX_MODEL_PATH = str(PROJECT_ROOT / "outputs" / "exports" / "efficientnet_b0_v9.onnx")
PYTORCH_CHECKPOINT = str(PROJECT_ROOT / "outputs" / "checkpoints" / "best_v9.pt")
MODEL_NAME = "efficientnet_b0"

# CLIP v9 (optional ensemble, loads from checkpoint if available)
CLIP_CHECKPOINT = str(PROJECT_ROOT / "outputs" / "checkpoints" / "best_v9_clip.pt")

# Inference settings
THRESHOLD = 0.65   # Raised from 0.5 → reduce False Positive (real→fake bias)
DEVICE = "cpu"

# Ensemble weights — EfficientNet-B0 dominates (more balanced on camera_real: 73.4%)
# CLIP was pulling too hard toward FAKE at 0.6 weight
EFFNET_WEIGHT = 0.6
CLIP_WEIGHT = 0.4

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
