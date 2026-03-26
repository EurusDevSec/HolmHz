"""
Model Service — ONNX Predictor + CLIP Predictor + Ensemble for HolmHz Web Demo.

OnnxPredictor: Fast EfficientNet inference using ONNX runtime.
CLIPPredictor: CLIP ViT-L/14 + linear probe for OOD robustness.
EnsemblePredictor: Combines both for best ID + OOD performance.
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

    def __init__(self, model_path: str, threshold: float = 0.5):
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


class CLIPPredictor:
    """Predict Real/Fake using CLIP ViT-L/14 + Linear Probe.

    Better OOD generalization than EfficientNet. Immune to
    JPEG compression artifacts (iPhone, Facebook, Instagram).
    """

    def __init__(self, checkpoint_path: str, threshold: float = 0.5, device: str = "cpu"):
        import torch
        from holmhz.utils.registry import DETECTOR_REGISTRY
        import holmhz.detectors  # noqa: F401

        self.device = device
        self.threshold = threshold

        model = DETECTOR_REGISTRY.build("clip_vit_l14", dropout=0.1, freeze_backbone=True)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

        # ImageNet normalization (same as CLIP default)
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    def preprocess(self, image: Image.Image) -> "torch.Tensor":
        """PIL Image -> [1, 3, 224, 224] torch tensor."""
        import torch

        img = image.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr).float()

    def predict(self, image: Image.Image) -> dict:
        """Predict Real/Fake with confidence."""
        import torch

        tensor = self.preprocess(image)
        with torch.no_grad():
            logit = self.model(tensor).item()
        prob_fake = float(1.0 / (1.0 + np.exp(-logit)))

        label = "FAKE" if prob_fake >= self.threshold else "REAL"
        confidence = prob_fake if label == "FAKE" else 1.0 - prob_fake

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "prob_fake": round(prob_fake, 4),
        }


class EnsemblePredictor:
    """Ensemble of EfficientNet (ONNX) + CLIP (PyTorch) + EXIF Hard Constraint.

    Strategy:
    1. Weighted average: p = w_effnet * p_effnet + w_clip * p_clip
    2. EXIF Hard Constraint: p *= exif_multiplier
       - Camera EXIF → 0.5x (strong REAL boost)
       - + GPS data → 0.85x additional
       - AI software → 1.2x (boost FAKE)
       - No EXIF → 1.0x (neutral — social media strips EXIF)
    3. Clamp to [0, 1]

    Fallback: If CLIP not available, uses EfficientNet only.
    """

    def __init__(
        self,
        effnet_predictor: OnnxPredictor,
        clip_predictor: CLIPPredictor = None,
        exif_analyzer=None,
        threshold: float = 0.5,
        effnet_weight: float = 0.4,
        clip_weight: float = 0.6,
    ):
        self.effnet = effnet_predictor
        self.clip = clip_predictor
        self.exif = exif_analyzer
        self.threshold = threshold
        self.w_effnet = effnet_weight
        self.w_clip = clip_weight

    def predict(self, image: Image.Image) -> dict:
        """Ensemble prediction with EXIF hard constraint.

        Returns:
            {label, confidence, prob_fake, prob_fake_raw,
             effnet_prob, clip_prob, model_used,
             exif_summary, exif_multiplier, exif_device}
        """
        # EfficientNet prediction (always available)
        effnet_result = self.effnet.predict(image)
        effnet_prob = effnet_result["prob_fake"]

        # CLIP prediction (if available)
        if self.clip is not None:
            clip_result = self.clip.predict(image)
            clip_prob = clip_result["prob_fake"]
            prob_fake_raw = self.w_effnet * effnet_prob + self.w_clip * clip_prob
            model_used = "ensemble"
        else:
            clip_prob = None
            prob_fake_raw = effnet_prob
            model_used = "efficientnet_only"

        # EXIF Hard Constraint
        exif_summary = "No EXIF analysis"
        exif_multiplier = 1.0
        exif_device = None

        if self.exif is not None:
            exif_result = self.exif.analyze(image)
            exif_summary = exif_result["exif_summary"]
            exif_multiplier = exif_result["multiplier"]
            exif_device = exif_result["device"]

        # Apply EXIF multiplier and clamp
        prob_fake = max(0.0, min(1.0, prob_fake_raw * exif_multiplier))

        label = "FAKE" if prob_fake >= self.threshold else "REAL"
        confidence = prob_fake if label == "FAKE" else 1.0 - prob_fake

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "prob_fake": round(prob_fake, 4),
            "prob_fake_raw": round(prob_fake_raw, 4),
            "effnet_prob": round(effnet_prob, 4),
            "clip_prob": round(clip_prob, 4) if clip_prob is not None else None,
            "model_used": model_used,
            "exif_summary": exif_summary,
            "exif_multiplier": round(exif_multiplier, 2),
            "exif_device": exif_device,
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
