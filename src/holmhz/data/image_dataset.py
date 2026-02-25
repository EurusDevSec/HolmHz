"""
Dataset class cho HolmHz.

Trong PyTorch, Dataset là "hợp đồng" (interface) định nghĩa:
  1. __len__(): Có bao nhiêu mẫu dữ liệu?
  2. __getitem__(index): Lấy mẫu thứ index ra.

DataLoader sẽ gọi 2 hàm này tự động:
  - Gọi __len__() để biết khi nào hết data (1 epoch)
  - Gọi __getitem__(0), __getitem__(1), ... để lấy từng mẫu
  - Tự động gom 32 mẫu thành 1 batch

Pattern:
  - CNNDetection: ImageFolder đơn giản (folder = label) — không biết source
  - DeepfakeBench: Abstract Dataset + nhiều subclass — quá phức tạp
  - HolmHz: JSON manifest (biết path + label + source, đơn giản)
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class ImageDataset(Dataset):
    """
    Dataset đọc ảnh từ manifest JSON file.

    Manifest format:
    [{"path": "...", "label": 0/1, "source": "ffhq", "category": "real"}, ...]

    Args:
        manifest_path: Đường dẫn tới file JSON manifest.
        transform: Albumentations transform pipeline.

    Example:
        >>> ds = ImageDataset("data/manifests/train.json", get_train_transforms())
        >>> sample = ds[0]
        >>> sample["image"].shape  # torch.Size([3, 224, 224])
        >>> sample["label"]        # tensor(0.) hoặc tensor(1.)
    """

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[A.Compose] = None,
    ):
        self.manifest_path = manifest_path
        self.transform = transform

        # Load manifest JSON
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data: list[dict] = json.load(f)

        if len(self.data) == 0:
            raise ValueError(f"Empty manifest: {manifest_path}")

        # Thống kê nhanh
        self.num_real = sum(1 for item in self.data if item["label"] == 0)
        self.num_fake = sum(1 for item in self.data if item["label"] == 1)

    def __len__(self) -> int:
        """Trả về tổng số ảnh trong dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """
        Trả về 1 mẫu dữ liệu.

        Flow:
        1. Đọc path và label từ manifest
        2. Load ảnh bằng OpenCV (nhanh hơn PIL cho augmentation)
        3. Chuyển BGR → RGB (OpenCV mặc định đọc BGR)
        4. Áp dụng transforms (augment + normalize + to tensor)
        5. Trả về dict {"image": tensor, "label": tensor, "source": str}

        Returns:
            dict với keys:
            - "image": tensor [3, 224, 224] (float32, normalized)
            - "label": tensor scalar (0.0 = real, 1.0 = fake)
            - "source": str (nguồn dữ liệu, ví dụ "cifake", "stylegan")
            - "path": str (đường dẫn ảnh gốc)
        """
        item = self.data[index]
        img_path = item["path"]

        # Load ảnh bằng OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")

        # BGR → RGB (OpenCV đọc BGR, Albumentations cần RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms (augment + normalize + to tensor)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]  # Đã là tensor [3, H, W] sau ToTensorV2()

        # Label → tensor float32 (cho BCEWithLogitsLoss)
        label = torch.tensor(item["label"], dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "source": item.get("source", "unknown"),
            "path": img_path,
        }

    def get_label_counts(self) -> dict[str, int]:
        """Trả về số lượng ảnh theo label."""
        return {"real": self.num_real, "fake": self.num_fake}

    def get_source_counts(self) -> dict[str, int]:
        """Trả về số lượng ảnh theo source."""
        from collections import Counter
        return dict(Counter(item["source"] for item in self.data))

    def __repr__(self) -> str:
        return (
            f"ImageDataset(manifest='{Path(self.manifest_path).name}', "
            f"total={len(self)}, real={self.num_real}, fake={self.num_fake})"
        )