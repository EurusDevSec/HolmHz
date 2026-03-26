"""
CLIP-based Detector — UniversalFakeDetect approach (CVPR 2023).

Architecture:
    Image → CLIP ViT-L/14 (frozen, 430M params) → 768-dim features → Linear(768→1) → logit

Why CLIP?
- CLIP learns SEMANTIC features (not pixel-level artifacts)
- Immune to JPEG compression / social media re-encoding
- SOTA generalization across unseen generators (SD3, DALL-E 3, etc.)
- Only 768 trainable params (linear probe)!

Reference:
- "Towards Universal Fake Image Detectors" (Ojha et al., CVPR 2023)
- UniversalFakeDetect GitHub: https://github.com/Yuheng-Li/UniversalFakeDetect
- "Raising the Bar of AI-generated Image Detection with CLIP" (2024)

Usage:
    model = CLIPDetector()  # CLIP ViT-L/14 + Linear(768→1)
    logits = model(images)  # images: [B, 3, 224, 224] (CLIP preprocessed)
"""

import torch
import torch.nn as nn

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False

from .base import BaseDetector


class CLIPDetector(BaseDetector):
    """
    CLIP ViT-L/14 + Linear Probe for AI image detection.

    Approach: Freeze CLIP backbone, train only a linear layer on top.
    This preserves CLIP's powerful semantic features while adapting
    to the fake/real classification task.

    Args:
        clip_model_name: CLIP model variant (default: "ViT-L-14")
        pretrained_dataset: Pre-training dataset (default: "openai")
        dropout: Dropout rate before classifier (default: 0.1)
        freeze_backbone: Always True for CLIP (default: True)

    Input: [B, 3, 224, 224] images (CLIP-preprocessed)
    Output: [B, 1] raw logits
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-L-14",
        pretrained_dataset: str = "openai",
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__()

        if not HAS_OPEN_CLIP:
            raise ImportError(
                "open_clip is required for CLIPDetector. "
                "Install: pip install open-clip-torch"
            )

        # Load CLIP model
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=pretrained_dataset,
        )
        self.clip_model.eval()

        # Get feature dimension
        self.feature_dim = self.clip_model.visual.output_dim  # 768 for ViT-L/14

        # Freeze CLIP backbone (always)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Classification head — only this is trained
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, 3, 224, 224] CLIP-preprocessed images

        Returns:
            [B, 1] raw logits (apply sigmoid for probability)
        """
        with torch.no_grad():
            features = self.clip_model.encode_image(x)  # [B, 768]
            features = features.float()  # CLIP outputs fp16 on GPU

        logits = self.head(features)  # [B, 1]
        return logits

    def get_feature_layer(self) -> nn.Module:
        """Return last transformer block for Grad-CAM (if needed)."""
        return self.clip_model.visual.transformer.resblocks[-1]

    @property
    def backbone(self):
        """For compatibility with existing training code."""
        return self.clip_model.visual

    def get_clip_preprocess(self):
        """Return CLIP's official preprocessing transform."""
        return self.preprocess
