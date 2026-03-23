"""
Grad-CAM Wrapper — Giải thích model nhìn vùng nào để phán đoán fake/real.

Sử dụng pytorch-grad-cam library. Tự động lấy target layer từ model.get_feature_layer().

Usage:
    from holmhz.xai.gradcam import GradCAMExplainer

    explainer = GradCAMExplainer(model, device="cuda")
    heatmap = explainer.explain(image_tensor)       # [H, W] float32 0-1
    overlay = explainer.overlay(image_tensor)        # [H, W, 3] uint8 BGR
    explainer.save(image_tensor, "output.png")
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMExplainer:
    """Grad-CAM wrapper for HolmHz detectors.

    Args:
        model: Detector with get_feature_layer() method
        device: torch device
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

        target_layer = model.get_feature_layer()
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def explain(self, image: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap for a single image.

        Args:
            image: [1, 3, H, W] normalized tensor

        Returns:
            [H, W] heatmap float32 in [0, 1]
        """
        image = image.to(self.device)
        grayscale_cam = self.cam(input_tensor=image)
        return grayscale_cam[0]  # [H, W]

    def overlay(
        self, image: torch.Tensor, rgb_image: np.ndarray
    ) -> np.ndarray:
        """Generate heatmap overlaid on original image.

        Args:
            image: [1, 3, H, W] normalized tensor (for model)
            rgb_image: [H, W, 3] float32 in [0, 1] (original image for display)

        Returns:
            [H, W, 3] uint8 overlay image
        """
        heatmap = self.explain(image)
        overlay = show_cam_on_image(rgb_image, heatmap, use_rgb=True)
        return overlay

    def save(
        self,
        image: torch.Tensor,
        rgb_image: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        """Generate and save Grad-CAM overlay to file.

        Args:
            image: [1, 3, H, W] normalized tensor
            rgb_image: [H, W, 3] float32 in [0, 1]
            output_path: Path to save the overlay image

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        overlay = self.overlay(image, rgb_image)
        # Convert RGB → BGR for cv2
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return output_path
