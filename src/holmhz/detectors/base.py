"""
Base class cho tất cả Detectors.

Detector = Backbone + Head (classification layers).
- Backbone "nhìn" ảnh → trích xuất đặc trưng (features)
- Head "phán đoán" → Real hay Fake

Tại sao tách Backbone và Head?
1. Swap backbone dễ (EfficientNet → CLIP)
2. Freeze/unfreeze backbone riêng
3. Grad-CAM cần truy cập backbone layers
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseDetector(ABC, nn.Module):
    """Abstract base class cho tất cả detectors.

    Mọi detector phải implement:
    - forward(x): ảnh → logits (raw scores, CHƯA sigmoid)

    Cung cấp sẵn:
    - predict(x): ảnh → labels (0 hoặc 1, đã qua sigmoid + threshold)
    - predict_proba(x): ảnh → probabilities (đã qua sigmoid)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ảnh → logits.

        Args:
            x: [B, 3, H, W] — batch ảnh đã normalize

        Returns:
            logits: [B, 1] — raw scores (CHƯA qua Sigmoid)
                Logit > 0 → nghiêng về Fake
                Logit < 0 → nghiêng về Real
                Logit = 0 → 50/50

        QUAN TRỌNG: Output là LOGITS, KHÔNG phải probabilities.
        - Training: dùng BCEWithLogitsLoss(logits, labels)
        - Inference: dùng predict_proba() để lấy P(Fake) ∈ [0,1]
        """

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Dự đoán nhãn (0 hoặc 1).

        Args:
            x: [B, 3, H, W] — batch ảnh
            threshold: ngưỡng phân loại (mặc định 0.5)

        Returns:
            labels: [B, 1] — 0 (Real) hoặc 1 (Fake)
        """
        probs = self.predict_proba(x)
        return (probs > threshold).long()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về probability P(Fake) ∈ [0, 1].

        Đã qua Sigmoid. Dùng khi inference (không phải training).

        Args:
            x: [B, 3, H, W] — batch ảnh

        Returns:
            probs: [B, 1] — P(Fake) cho mỗi ảnh
                0.0 = chắc chắn Real
                1.0 = chắc chắn Fake
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
