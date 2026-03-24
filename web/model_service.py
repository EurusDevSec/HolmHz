"""
Model Service — ONNX Predictor + Grad-CAM Service for HolmHz Web Demo.

OnnxPredictor: Fast inference using onnxruntime (no PyTorch needed for predict).
GradCAMService: Heatmap generation using PyTorch + pytorch-grad-cam.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add src/ to path for holmhz imports
_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


class OnnxPredictor:
    """Predict Real/Fake using ONNX model.

    Uses onnxruntime for fast CPU inference (~100ms).
    No PyTorch dependency needed.
    """

    def __init__(self, model_path: str, threshold: float = 0.76):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.threshold = threshold
        self.input_name = self.session.get_inputs()[0].name

        # ImageNet normalization arrays
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """PIL Image -> [1, 3, 224, 224] normalized float32 array."""
        img = image.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
        arr = (arr - self.mean) / self.std
        return arr

    def predict(self, image: Image.Image) -> dict:
        """Predict Real/Fake with confidence.

        Returns:
            {label: "FAKE"|"REAL", confidence: float, prob_fake: float}
        """
        input_arr = self.preprocess(image)
        logit = self.session.run(None, {self.input_name: input_arr})[0][0][0]
        prob_fake = float(1.0 / (1.0 + np.exp(-float(logit))))

        label = "FAKE" if prob_fake >= self.threshold else "REAL"
        confidence = prob_fake if label == "FAKE" else 1.0 - prob_fake

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "prob_fake": round(prob_fake, 4),
        }


class GradCAMService:
    """Generate Grad-CAM heatmap overlays.

    Requires PyTorch + pytorch-grad-cam.
    Loads model once on init, reuses for all requests.
    """

    def __init__(self, model_name: str, checkpoint_path: str, device: str = "cpu"):
        import torch
        from holmhz.utils.registry import DETECTOR_REGISTRY
        import holmhz.detectors  # noqa: F401

        self.device = device
        model = DETECTOR_REGISTRY.build(model_name, pretrained=False, freeze_backbone=False)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

        from holmhz.xai.gradcam import GradCAMExplainer
        self.explainer = GradCAMExplainer(model, device=device)

    def generate_heatmap(self, image: Image.Image) -> Image.Image:
        """Generate Grad-CAM overlay for input image.

        Args:
            image: PIL Image (any size, will be resized)

        Returns:
            PIL Image with heatmap overlay (224x224)
        """
        import tempfile
        import os
        from holmhz.xai.utils import load_image_for_gradcam

        # Save PIL -> temp file -> load with gradcam utils
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            image.save(temp_path)

        try:
            tensor, rgb_image = load_image_for_gradcam(temp_path)
            overlay = self.explainer.overlay(tensor, rgb_image)
            return Image.fromarray(overlay)
        finally:
            os.unlink(temp_path)
