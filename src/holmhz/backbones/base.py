"""
Base class cho tất cả Backbones.

Backbone = phần "mắt" của model — biết nhìn ảnh, trích xuất đặc trưng.

Tại sao cần Abstract Base Class?
→ Định nghĩa "hợp đồng": mọi backbone PHẢI có:
  - extract_features(x) → vector features
  - get_features_dim() → số chiều features
→ Đổi backbone: code khác KHÔNG cần sửa (Open/Closed Principle).

Pattern từ DeepfakeBench: AbstractDetector định nghĩa interface chung.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(ABC, nn.Module):
    """Abstract base class cho tất cả backbones.

    Mọi backbone kế thừa class này phải implement:
    - extract_features(): trích xuất feature vector từ ảnh
    - get_features_dim(): trả về số chiều của feature vector

    Cung cấp sẵn:
    - freeze(): đóng băng tất cả params
    - unfreeze(): mở khóa tất cả params
    - forward(): alias cho extract_features (nn.Module compatibility)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất features từ ảnh input.

        Args:
            x: Tensor [B, 3, H, W] — batch ảnh đã normalize

        Returns:
            features: Tensor [B, features_dim] — vector đặc trưng
        """

    @abstractmethod
    def get_features_dim(self) -> int:
        """Trả về kích thước vector features.

        Ví dụ: 1280 cho EfficientNet-B0, 768 cho CLIP ViT-B.
        """

    def freeze(self) -> None:
        """Đóng băng tất cả parameters — không cho gradient chạy qua.

        Dùng trong Phase 1 Transfer Learning:
        backbone.freeze() → chỉ train head.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Mở khóa tất cả parameters — cho phép training.

        Dùng trong Phase 2 Fine-tuning:
        backbone.unfreeze() → train toàn bộ model.
        """
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass = extract_features.

        nn.Module yêu cầu forward(), nhưng logic thực sự ở extract_features().
        Giữ forward() để có thể dùng backbone(x) thay vì backbone.extract_features(x).
        """
        return self.extract_features(x)
