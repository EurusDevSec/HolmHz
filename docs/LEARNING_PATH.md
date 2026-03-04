# 📚 LEARNING PATH - HolmHz Project

> **Lộ trình học tập Hands-on cho dự án Phát hiện ảnh tổng hợp**  
> Dành cho: Lê Văn Hoàng  
> Mục tiêu: Vừa học kiến thức vừa xây dựng dự án HolmHz  
> Thời gian: Song song với timeline dự án (11/2025 - 05/2026)

---

## 📋 Mục lục

1. [Tổng quan lộ trình](#1-tổng-quan-lộ-trình)
2. [Tuần 1-2: Python & Deep Learning Foundations](#2-tuần-1-2-python--deep-learning-foundations)
3. [Tuần 3-4: PyTorch Fundamentals](#3-tuần-3-4-pytorch-fundamentals)
4. [Tuần 5-6: Image Classification & CNN](#4-tuần-5-6-image-classification--cnn)
5. [Tuần 7-8: Transfer Learning & EfficientNet](#5-tuần-7-8-transfer-learning--efficientnet)
6. [Tuần 9-10: Deepfake Detection Specifics](#6-tuần-9-10-deepfake-detection-specifics)
7. [Tuần 11-12: XAI & Grad-CAM](#7-tuần-11-12-xai--grad-cam)
8. [Tuần 13-16: Web Application](#8-tuần-13-16-web-application)
9. [Tài nguyên bổ sung](#9-tài-nguyên-bổ-sung)

---

## 1. Tổng quan lộ trình

### 1.1. Bạn cần học gì cho HolmHz?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE MAP CHO HOLMHZ                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LEVEL 1: Nền tảng (Tuần 1-4)                                          │
│  ─────────────────────────────                                          │
│  ├── Python nâng cao (OOP, decorators, type hints)                     │
│  ├── NumPy & Pandas (xử lý data)                                       │
│  ├── PyTorch cơ bản (tensor, autograd, nn.Module)                      │
│  └── Jupyter/Colab workflow                                            │
│                                                                         │
│  LEVEL 2: Computer Vision (Tuần 5-8)                                   │
│  ────────────────────────────────────                                   │
│  ├── Image Processing (PIL, OpenCV, augmentation)                      │
│  ├── CNN Architecture (Conv, Pool, BatchNorm)                          │
│  ├── Transfer Learning (pretrained models, fine-tuning)                │
│  └── EfficientNet (architecture, timm library)                         │
│                                                                         │
│  LEVEL 3: Deepfake Detection (Tuần 9-12)                               │
│  ───────────────────────────────────────                                │
│  ├── GAN & Diffusion basics (biết cách chúng tạo ảnh)                  │
│  ├── Deepfake artifacts (lỗi mà AI tạo ra)                             │
│  ├── Frequency domain analysis (DCT, FFT - optional)                   │
│  ├── Grad-CAM (explainable AI)                                         │
│  └── Evaluation metrics (AUC, ROC, confusion matrix)                   │
│                                                                         │
│  LEVEL 4: Deployment (Tuần 13-16)                                      │
│  ──────────────────────────────                                         │
│  ├── FastAPI (REST API)                                                │
│  ├── Gradio (UI nhanh)                                                 │
│  ├── ONNX (model optimization)                                         │
│  └── Basic Docker (optional)                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2. Nguyên tắc học

| Nguyên tắc          | Mô tả                                          |
| ------------------- | ---------------------------------------------- |
| **Learn by Doing**  | Mỗi khái niệm → code ngay → áp dụng vào HolmHz |
| **Just-in-Time**    | Học đúng thứ cần, đúng lúc cần                 |
| **80/20 Rule**      | Focus 20% kiến thức quan trọng nhất            |
| **Build in Public** | Commit code thường xuyên, ghi chú học được gì  |

---

## 2. Tuần 1-2: Python & Deep Learning Foundations

### 2.1. Mục tiêu

- [ ] Hiểu OOP trong Python (class, inheritance, dunder methods)
- [ ] Sử dụng thành thạo NumPy array operations
- [ ] Hiểu cơ bản về Neural Networks (forward, backward, gradient)
- [ ] Setup môi trường dev (VS Code, uv/pip, Git)

### 2.2. Tài liệu học

| Tài liệu                          | Link                                                                                         | Thời gian | Priority |
| --------------------------------- | -------------------------------------------------------------------------------------------- | --------- | -------- |
| **Python OOP Crash Course**       | [YouTube - Corey Schafer](https://www.youtube.com/watch?v=ZDa-Z5JzLYM)                       | 1h        | 🔴       |
| **NumPy in 1 Hour**               | [freeCodeCamp](https://www.youtube.com/watch?v=QUT1VHiLmmI)                                  | 1h        | 🔴       |
| **3Blue1Brown - Neural Networks** | [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | 1h        | 🔴       |
| Deep Learning Book Ch.1-6         | [deeplearningbook.org](https://www.deeplearningbook.org/)                                    | Optional  | 🟡       |

### 2.3. Hands-on Exercises (Làm trong HolmHz)

#### Exercise 2.1: Setup Project Structure

```python
# Tạo file: src/utils/helpers.py
# Mục tiêu: Thực hành OOP và type hints

from typing import List, Tuple, Optional
from pathlib import Path

class ImagePathManager:
    """Quản lý đường dẫn ảnh trong dataset."""

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self._validate_root()

    def _validate_root(self) -> None:
        """Kiểm tra thư mục tồn tại."""
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")

    def get_image_paths(self, extension: str = "*.jpg") -> List[Path]:
        """Lấy tất cả đường dẫn ảnh."""
        return list(self.root.rglob(extension))

    def split_by_label(self) -> Tuple[List[Path], List[Path]]:
        """Tách ảnh theo label (real/fake)."""
        real = list(self.root.glob("real/*"))
        fake = list(self.root.glob("fake/*"))
        return real, fake

# Test
if __name__ == "__main__":
    manager = ImagePathManager("data/processed/train")
    print(f"Found {len(manager.get_image_paths())} images")
```

**📝 Ghi chú học được**:

- `Path` từ pathlib tốt hơn string cho file paths
- Type hints (`str`, `List[Path]`) giúp code dễ đọc
- Dunder method `__init__` là constructor

#### Exercise 2.2: NumPy Image Operations

```python
# Tạo file: notebooks/01_numpy_basics.ipynb
# Mục tiêu: Hiểu image như array

import numpy as np
from PIL import Image

# Load ảnh thành numpy array
img = Image.open("sample.jpg")
arr = np.array(img)

print(f"Shape: {arr.shape}")  # (H, W, C) = (Height, Width, Channels)
print(f"Dtype: {arr.dtype}")  # uint8 (0-255)
print(f"Min/Max: {arr.min()}, {arr.max()}")

# Normalize về [0, 1]
arr_normalized = arr / 255.0
print(f"After normalize: {arr_normalized.min()}, {arr_normalized.max()}")

# ImageNet normalization (rất quan trọng cho pretrained models!)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
arr_imagenet = (arr_normalized - mean) / std

# 📝 Tại sao cần normalize?
# - Pretrained models (EfficientNet) được train với ImageNet stats
# - Nếu không normalize, model sẽ cho kết quả sai
```

**📝 Ghi chú học được**:

- Ảnh RGB = array shape (H, W, 3)
- Giá trị pixel: 0-255 (uint8) hoặc 0-1 (float)
- ImageNet normalization là PHẢI CÓ khi dùng pretrained models

### 2.4. Kiểm tra hiểu biết

- [ ] Giải thích được `self` trong Python class
- [ ] Biết cách reshape numpy array từ (H, W, C) sang (C, H, W)
- [ ] Hiểu tại sao cần normalize ảnh trước khi đưa vào neural network

---

## 3. Tuần 3-4: PyTorch Fundamentals

### 3.1. Mục tiêu

- [ ] Hiểu Tensor và sự khác biệt với NumPy array
- [ ] Hiểu Autograd (automatic differentiation)
- [ ] Viết được nn.Module đơn giản
- [ ] Hiểu training loop (forward → loss → backward → update)

### 3.2. Tài liệu học

| Tài liệu                      | Link                                                                                       | Thời gian    | Priority |
| ----------------------------- | ------------------------------------------------------------------------------------------ | ------------ | -------- |
| **PyTorch in 60 Minutes**     | [Official Tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | 1h           | 🔴       |
| **PyTorch for Deep Learning** | [freeCodeCamp 25h course](https://www.youtube.com/watch?v=V_xro1bcAuA)                     | Xem phần 1-5 | 🔴       |
| Learn PyTorch                 | [learnpytorch.io](https://www.learnpytorch.io/)                                            | Reference    | 🟡       |

### 3.3. Hands-on Exercises

#### Exercise 3.1: Tensor Basics

```python
# Tạo file: notebooks/02_pytorch_basics.ipynb

import torch
import numpy as np

# Tạo tensor
x = torch.randn(3, 224, 224)  # Random tensor giống shape ảnh (C, H, W)
print(f"Shape: {x.shape}, Device: {x.device}, Dtype: {x.dtype}")

# Chuyển từ NumPy
arr = np.random.rand(224, 224, 3).astype(np.float32)
tensor = torch.from_numpy(arr)
# Đổi từ (H, W, C) → (C, H, W) cho PyTorch
tensor = tensor.permute(2, 0, 1)

# GPU acceleration (nếu có)
if torch.cuda.is_available():
    x = x.cuda()  # hoặc x.to('cuda')
    print(f"Now on: {x.device}")

# 📝 PyTorch yêu cầu:
# - Shape ảnh: (Batch, Channels, Height, Width) = (B, C, H, W)
# - Khác với NumPy/PIL: (H, W, C)
```

#### Exercise 3.2: Simple Neural Network

```python
# File: src/models/simple_classifier.py
# Mục tiêu: Hiểu nn.Module

import torch
import torch.nn as nn

class SimpleBinaryClassifier(nn.Module):
    """
    Classifier đơn giản cho binary classification (Real/Fake).
    Đây là version simplified - sau sẽ thay bằng EfficientNet.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 1):
        super().__init__()

        # Feature extractor (CNN layers)
        self.features = nn.Sequential(
            # Conv Block 1: 3 → 32 channels
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 → 112

            # Conv Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 → 56

            # Conv Block 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling → (B, 128, 1, 1)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor shape (B, 3, 224, 224)

        Returns:
            Logits shape (B, 1)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

# Test
if __name__ == "__main__":
    model = SimpleBinaryClassifier()

    # Fake input batch
    batch = torch.randn(4, 3, 224, 224)  # 4 images

    output = model(batch)
    print(f"Input: {batch.shape} → Output: {output.shape}")
    # Expected: torch.Size([4, 3, 224, 224]) → torch.Size([4, 1])

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
```

**📝 Ghi chú học được**:

- `nn.Module` là base class cho mọi model PyTorch
- `__init__`: định nghĩa layers
- `forward`: định nghĩa data flow
- `nn.Sequential`: gom nhiều layers thành 1 block
- `BatchNorm` giúp training ổn định hơn

#### Exercise 3.3: Training Loop

```python
# File: notebooks/03_training_loop.ipynb
# Mục tiêu: Hiểu training loop hoàn chỉnh

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Tạo fake dataset
X = torch.randn(100, 3, 224, 224)  # 100 fake images
y = torch.randint(0, 2, (100, 1)).float()  # Labels: 0 or 1
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 2. Model, Loss, Optimizer
model = SimpleBinaryClassifier()
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Training Loop
num_epochs = 3

for epoch in range(num_epochs):
    model.train()  # Set training mode
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 📝 Training loop anatomy:
# 1. Forward: model(x) → predictions
# 2. Loss: criterion(predictions, labels) → scalar
# 3. Backward: loss.backward() → compute gradients
# 4. Update: optimizer.step() → update weights
# 5. Zero grad: optimizer.zero_grad() → clear for next iteration
```

### 3.4. Kiểm tra hiểu biết

- [ ] Giải thích sự khác biệt giữa `model.train()` và `model.eval()`
- [ ] Tại sao cần `optimizer.zero_grad()` trước mỗi backward?
- [ ] `BCEWithLogitsLoss` khác gì `BCELoss`?

---

## 4. Tuần 5-6: Image Classification & CNN

### 4.1. Mục tiêu

- [ ] Hiểu kiến trúc CNN (Conv, Pool, FC)
- [ ] Sử dụng thành thạo PIL và Albumentations
- [ ] Viết được custom Dataset class
- [ ] Hiểu data augmentation và tại sao cần

### 4.2. Tài liệu học

| Tài liệu                    | Link                                                                                | Thời gian | Priority |
| --------------------------- | ----------------------------------------------------------------------------------- | --------- | -------- |
| **CS231n CNN**              | [Stanford Notes](https://cs231n.github.io/convolutional-networks/)                  | 2h        | 🔴       |
| **Albumentations Tutorial** | [Official Docs](https://albumentations.ai/docs/getting_started/image_augmentation/) | 1h        | 🔴       |
| CNN Explainer               | [Interactive](https://poloclub.github.io/cnn-explainer/)                            | 30m       | 🟢       |

### 4.3. Hands-on Exercises

#### Exercise 4.1: Custom Dataset cho HolmHz

```python
# File: src/data/dataset.py
# Đây là file THỰC SỰ dùng cho project

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, List
import json

class SyntheticImageDataset(Dataset):
    """
    Dataset cho bài toán phát hiện ảnh tổng hợp.

    Structure expected:
        data/processed/
        ├── train/
        │   ├── real/
        │   │   ├── ffhq_00001.jpg
        │   │   └── ...
        │   └── fake/
        │       ├── stylegan_00001.jpg
        │       └── ...
        └── manifest.json  # Optional: metadata
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        return_metadata: bool = False,
    ):
        """
        Args:
            root_dir: Đường dẫn đến data/processed
            split: "train", "val", hoặc "test"
            transform: Albumentations transform
            return_metadata: Có trả về source info không
        """
        self.root = Path(root_dir) / split
        self.transform = transform or self._default_transform()
        self.return_metadata = return_metadata

        # Load image paths
        self.samples: List[Tuple[Path, int, str]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load all image paths with labels."""
        # Real images (label = 0)
        real_dir = self.root / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*.jpg"):
                source = self._extract_source(img_path)
                self.samples.append((img_path, 0, source))

        # Fake images (label = 1)
        fake_dir = self.root / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*.jpg"):
                source = self._extract_source(img_path)
                self.samples.append((img_path, 1, source))

        print(f"Loaded {len(self.samples)} samples from {self.root}")

    def _extract_source(self, path: Path) -> str:
        """Extract source from filename (e.g., 'stylegan_00001.jpg' → 'stylegan')."""
        return path.stem.split("_")[0]

    def _default_transform(self) -> A.Compose:
        """Default transform nếu không specify."""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, source = self.samples[idx]

        # Load image
        image = Image.open(path).convert("RGB")
        image = np.array(image)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Return
        if self.return_metadata:
            return image, label, {"source": source, "path": str(path)}
        return image, label


def get_train_transforms() -> A.Compose:
    """Augmentation cho training - MÔ PHỎNG điều kiện thực tế."""
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),

        # Mô phỏng ảnh bị nén, blur, noise trên internet
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.GaussNoise(var_limit=(10, 50), p=1),
        ], p=0.3),

        # Color jittering
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.3,
        ),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Không augmentation cho validation/test."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# Test
if __name__ == "__main__":
    import numpy as np

    # Tạo fake data để test
    from pathlib import Path

    dataset = SyntheticImageDataset(
        root_dir="data/processed",
        split="train",
        transform=get_train_transforms(),
    )

    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Image shape: {img.shape}, Label: {label}")
```

**📝 Ghi chú học được**:

- Custom Dataset cần implement: `__init__`, `__len__`, `__getitem__`
- Albumentations nhanh hơn torchvision transforms
- Training cần augmentation, validation/test KHÔNG cần
- JPEG compression augmentation quan trọng vì ảnh trên web thường bị nén

#### Exercise 4.2: Visualize Augmentations

```python
# File: notebooks/04_visualize_augmentation.ipynb
# Mục tiêu: Hiểu augmentation làm gì

import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
import numpy as np

# Load sample image
img = Image.open("sample_face.jpg")
img_array = np.array(img)

# Define augmentation
aug = A.Compose([
    A.HorizontalFlip(p=1),
    A.ImageCompression(quality_lower=40, quality_upper=60, p=1),
    A.GaussianBlur(blur_limit=7, p=1),
])

# Visualize multiple augmentations
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(img_array)
axes[0, 0].set_title("Original")

for i in range(1, 8):
    row, col = i // 4, i % 4
    augmented = aug(image=img_array)["image"]
    axes[row, col].imshow(augmented)
    axes[row, col].set_title(f"Aug {i}")

plt.tight_layout()
plt.savefig("outputs/augmentation_examples.png")

# 📝 Tại sao augmentation quan trọng cho deepfake detection?
# 1. Ảnh trên internet bị JPEG compress → mô phỏng bằng ImageCompression
# 2. Ảnh bị resize khi share → mô phỏng bằng RandomScale
# 3. Ảnh bị screenshot, blur → mô phỏng bằng GaussianBlur
# → Model học được features robust hơn
```

### 4.4. Kiểm tra hiểu biết

- [ ] Conv2d(3, 32, kernel_size=3) có bao nhiêu parameters?
- [ ] Tại sao cần augmentation khi training?
- [ ] `ToTensorV2()` làm gì với shape của ảnh?

---

## 5. Tuần 7-8: Transfer Learning & EfficientNet

### 5.1. Mục tiêu

- [ ] Hiểu Transfer Learning và tại sao nó hiệu quả
- [ ] Sử dụng thành thạo thư viện timm
- [ ] Fine-tune EfficientNet cho binary classification
- [ ] Hiểu freezing/unfreezing layers

### 5.2. Tài liệu học

| Tài liệu                       | Link                                                                                       | Thời gian | Priority |
| ------------------------------ | ------------------------------------------------------------------------------------------ | --------- | -------- |
| **Transfer Learning Tutorial** | [PyTorch Official](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | 1h        | 🔴       |
| **timm Library**               | [GitHub](https://github.com/huggingface/pytorch-image-models)                              | 1h        | 🔴       |
| **EfficientNet Paper**         | [arXiv](https://arxiv.org/abs/1905.11946)                                                  | Optional  | 🟡       |

### 5.3. Hands-on Exercises

#### Exercise 5.1: EfficientNet với timm

```python
# File: src/models/efficientnet.py
# Đây là model CHÍNH của dự án HolmHz

import torch
import torch.nn as nn
import timm
from typing import Optional

class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 cho binary classification (Real/Fake).

    Sử dụng pretrained weights từ ImageNet.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features  # 1280 for B0

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        # Freeze backbone initially (optional)
        # self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze backbone weights - chỉ train classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze backbone - train toàn bộ model."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, 224, 224)

        Returns:
            Logits (B, 1)
        """
        # Extract features
        features = self.backbone(x)  # (B, 1280)

        # Classify
        logits = self.classifier(features)  # (B, 1)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings (useful for analysis)."""
        return self.backbone(x)


def create_model(
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
) -> EfficientNetClassifier:
    """
    Factory function để tạo model.

    Args:
        model_name: Tên model từ timm
        pretrained: Dùng ImageNet weights
        checkpoint_path: Load từ checkpoint đã train

    Returns:
        Initialized model
    """
    model = EfficientNetClassifier(
        model_name=model_name,
        pretrained=pretrained,
    )

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model


# Test
if __name__ == "__main__":
    model = create_model()

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # List available models
    print("\nAvailable EfficientNet variants:")
    for name in timm.list_models("efficientnet*"):
        print(f"  - {name}")
```

**📝 Ghi chú học được**:

- `timm.create_model()` tải pretrained model
- `num_classes=0` để loại bỏ head, lấy features
- EfficientNet-B0 có 1280-dim features
- Freeze backbone giúp train nhanh hơn, ít overfit

#### Exercise 5.2: Training Script Hoàn Chỉnh

```python
# File: src/training/trainer.py
# Training loop đầy đủ cho HolmHz

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
from typing import Dict, Optional
from pathlib import Path

class Trainer:
    """Trainer class cho binary classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 20),
            eta_min=1e-6,
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.best_val_auc = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """Train 1 epoch."""
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)

            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": loss.item()})

        # Compute metrics
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_preds)

        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_auc": auc,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import roc_auc_score, accuracy_score
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_auc": auc,
            "val_acc": acc,
        }

    def save_checkpoint(self, path: str, metrics: Dict) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        print(f"Saved checkpoint to {path}")

    def fit(self, epochs: int, save_dir: str = "outputs/checkpoints") -> None:
        """Full training loop."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*50}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Log
            all_metrics = {**train_metrics, **val_metrics, "lr": self.scheduler.get_last_lr()[0]}
            print(f"Metrics: {all_metrics}")

            # Log to wandb
            if wandb.run:
                wandb.log(all_metrics)

            # Save best model
            if val_metrics["val_auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["val_auc"]
                self.save_checkpoint(
                    f"{save_dir}/best_model.pt",
                    val_metrics,
                )

        print(f"\nTraining complete! Best AUC: {self.best_val_auc:.4f}")
```

### 5.4. Kiểm tra hiểu biết

- [ ] Transfer learning hoạt động như thế nào?
- [ ] Khi nào nên freeze backbone, khi nào nên unfreeze?
- [ ] Tại sao dùng AdamW thay vì Adam?

---

## 6. Tuần 9-10: Deepfake Detection Specifics

### 6.1. Mục tiêu

- [ ] Hiểu GAN và Diffusion models tạo ảnh như thế nào
- [ ] Nhận biết artifacts trong ảnh AI-generated
- [ ] Hiểu tại sao cross-dataset generalization khó
- [ ] Đọc hiểu papers liên quan

### 6.2. Tài liệu học

| Tài liệu                       | Link                                                                    | Thời gian | Priority |
| ------------------------------ | ----------------------------------------------------------------------- | --------- | -------- |
| **GAN Basics**                 | [3Blue1Brown-style video](https://www.youtube.com/watch?v=-Upj_VhjTBs)  | 30m       | 🔴       |
| **Stable Diffusion Explained** | [Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/) | 1h        | 🔴       |
| **Wang et al. (2020) Paper**   | [arXiv](https://arxiv.org/abs/1912.11035)                               | 1h        | 🔴       |
| Awesome Deepfakes Detection    | [GitHub](https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection)    | Reference | 🟡       |

### 6.3. Hands-on Exercises

#### Exercise 6.1: Phân tích Deepfake Artifacts

```python
# File: notebooks/05_analyze_artifacts.ipynb
# Mục tiêu: Hiểu các lỗi trong ảnh AI-generated

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def load_and_compare(real_path: str, fake_path: str):
    """So sánh ảnh thật và giả side-by-side."""
    real = np.array(Image.open(real_path))
    fake = np.array(Image.open(fake_path))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original images
    axes[0, 0].imshow(real)
    axes[0, 0].set_title("Real Image")
    axes[0, 1].imshow(fake)
    axes[0, 1].set_title("Fake Image (AI-generated)")

    # Difference (nếu cùng pose)
    # axes[0, 2].imshow(np.abs(real.astype(int) - fake.astype(int)))
    # axes[0, 2].set_title("Difference")

    # Row 2: Zoom vào vùng hay có lỗi
    # Mắt
    eye_region_real = real[100:180, 80:160]
    eye_region_fake = fake[100:180, 80:160]
    axes[1, 0].imshow(eye_region_real)
    axes[1, 0].set_title("Real - Eye Region")
    axes[1, 1].imshow(eye_region_fake)
    axes[1, 1].set_title("Fake - Eye Region (Check symmetry)")

    # Edge detection
    real_edges = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), 100, 200)
    fake_edges = cv2.Canny(cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY), 100, 200)
    axes[1, 2].imshow(fake_edges, cmap='gray')
    axes[1, 2].set_title("Fake - Edge Detection")

    plt.tight_layout()
    plt.savefig("outputs/artifact_analysis.png")

# 📝 Các artifacts thường gặp trong ảnh AI-generated:
#
# 1. MẮT:
#    - Không đối xứng
#    - Phản chiếu ánh sáng không nhất quán
#    - Pupils hình dạng lạ
#
# 2. TÓC:
#    - Texture không tự nhiên
#    - Boundary với background mờ
#
# 3. RĂNG:
#    - Số lượng không đúng
#    - Shape không tự nhiên
#
# 4. TAI:
#    - Không đối xứng
#    - Earrings không khớp
#
# 5. BACKGROUND:
#    - Blending artifacts
#    - Inconsistent lighting
#
# 6. FREQUENCY ARTIFACTS:
#    - GAN: up-sampling tạo patterns trong frequency domain
#    - Diffusion: khác biệt trong noise distribution
```

#### Exercise 6.2: Frequency Domain Analysis

```python
# File: notebooks/06_frequency_analysis.ipynb
# Mục tiêu: Hiểu tại sao frequency domain hữu ích

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, fftshift

def analyze_frequency(image_path: str, title: str):
    """Phân tích ảnh trong frequency domain."""
    # Load và convert to grayscale
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # FFT
    f_transform = fft2(img_array)
    f_shift = fftshift(f_transform)
    magnitude = np.log(np.abs(f_shift) + 1)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title(f"{title} - Spatial Domain")

    axes[1].imshow(magnitude, cmap='hot')
    axes[1].set_title(f"{title} - Frequency Spectrum")

    return magnitude

# Compare real vs fake
real_spectrum = analyze_frequency("samples/real.jpg", "Real")
fake_spectrum = analyze_frequency("samples/stylegan.jpg", "StyleGAN")

# 📝 Tại sao frequency analysis hoạt động?
#
# 1. GAN ARTIFACTS:
#    - GAN sử dụng up-sampling (nearest neighbor, bilinear)
#    - Up-sampling tạo ra periodic patterns trong frequency domain
#    - Thể hiện như các "peaks" lặp lại trong spectrum
#
# 2. DIFFUSION ARTIFACTS:
#    - Diffusion models có noise distribution khác ảnh thật
#    - High-frequency components khác biệt
#
# 3. JPEG COMPRESSION:
#    - JPEG cũng tạo artifacts trong frequency domain (8x8 blocks)
#    - Cần phân biệt với AI artifacts
#
# → Đây là lý do tại sao plan.md yêu cầu Frequency Branch!
```

### 6.4. Kiểm tra hiểu biết

- [ ] GAN và Diffusion tạo ảnh khác nhau như thế nào?
- [ ] Tại sao model train trên StyleGAN có thể fail với SDXL?
- [ ] Frequency artifacts trong ảnh GAN là gì?

---

## 7. Tuần 11-12: XAI & Grad-CAM

### 7.1. Mục tiêu

- [ ] Hiểu Explainable AI (XAI) và tầm quan trọng
- [ ] Implement và sử dụng Grad-CAM
- [ ] Tạo heatmap visualization đẹp
- [ ] Validate model "nhìn" đúng vùng

### 7.2. Tài liệu học

| Tài liệu             | Link                                                                                             | Thời gian | Priority |
| -------------------- | ------------------------------------------------------------------------------------------------ | --------- | -------- |
| **Grad-CAM Paper**   | [arXiv](https://arxiv.org/abs/1610.02391)                                                        | 1h        | 🔴       |
| **pytorch-grad-cam** | [GitHub](https://github.com/jacobgil/pytorch-grad-cam)                                           | 1h        | 🔴       |
| CAM Explained        | [Blog](https://glassboxmedicine.com/2020/05/29/grad-cam-visual-explanations-from-deep-networks/) | 30m       | 🟢       |

### 7.3. Hands-on Exercises

#### Exercise 7.1: Grad-CAM cho HolmHz

```python
# File: src/xai/gradcam.py
# XAI module cho dự án

import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class ExplainabilityEngine:
    """
    Engine để giải thích predictions bằng Grad-CAM.

    Giúp visualize vùng nào trên ảnh khiến model quyết định Real/Fake.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Optional[List] = None,
    ):
        """
        Args:
            model: Trained model
            target_layers: Layers để compute CAM. Nếu None, tự detect.
        """
        self.model = model
        self.model.eval()

        # Auto-detect target layer (last conv layer)
        if target_layers is None:
            # Cho EfficientNet, target là layer cuối của backbone
            target_layers = [self._get_last_conv_layer()]

        self.target_layers = target_layers

        # Initialize Grad-CAM
        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers,
        )

    def _get_last_conv_layer(self):
        """Tự động tìm conv layer cuối cùng."""
        # Cho EfficientNet từ timm
        # backbone.conv_head hoặc backbone.bn2
        if hasattr(self.model, 'backbone'):
            if hasattr(self.model.backbone, 'conv_head'):
                return self.model.backbone.conv_head
            # Fallback
            for name, module in self.model.backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            return last_conv
        raise ValueError("Cannot auto-detect target layer")

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: int = 1,  # 1 = Fake
    ) -> np.ndarray:
        """
        Generate CAM heatmap cho một ảnh.

        Args:
            image_tensor: Preprocessed image (1, 3, 224, 224)
            target_class: 0 = Real, 1 = Fake

        Returns:
            Grayscale CAM (224, 224)
        """
        targets = [BinaryClassifierOutputTarget(target_class)]

        grayscale_cam = self.cam(
            input_tensor=image_tensor,
            targets=targets,
        )

        return grayscale_cam[0]  # (224, 224)

    def visualize(
        self,
        image_path: str,
        transform,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Full pipeline: load → predict → explain → visualize.

        Returns:
            (visualization, probability, prediction)
        """
        # Load original image
        original = Image.open(image_path).convert("RGB")
        original_np = np.array(original.resize((224, 224))) / 255.0

        # Preprocess
        transformed = transform(image=np.array(original))
        input_tensor = transformed["image"].unsqueeze(0)

        # Predict
        with torch.no_grad():
            logits = self.model(input_tensor)
            prob = torch.sigmoid(logits).item()

        prediction = "FAKE" if prob > 0.5 else "REAL"

        # Explain
        cam = self.explain(input_tensor, target_class=1)

        # Overlay
        visualization = show_cam_on_image(
            original_np.astype(np.float32),
            cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
        )

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(visualization)
        axes[2].set_title(f"{prediction} ({prob:.1%})")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return visualization, prob, prediction


# Demo usage
if __name__ == "__main__":
    from src.models.efficientnet import create_model
    from src.data.dataset import get_val_transforms

    # Load model
    model = create_model(checkpoint_path="outputs/checkpoints/best_model.pt")

    # Create explainer
    explainer = ExplainabilityEngine(model)

    # Explain a sample
    viz, prob, pred = explainer.visualize(
        image_path="samples/test_fake.jpg",
        transform=get_val_transforms(),
        save_path="outputs/gradcam_example.png",
    )

    print(f"Prediction: {pred} ({prob:.2%} confidence)")
```

**📝 Ghi chú học được**:

- Grad-CAM cho thấy vùng nào quan trọng cho quyết định
- Target layer thường là conv layer cuối cùng
- Heatmap đỏ = quan trọng, xanh = không quan trọng
- Validate: heatmap nên highlight vùng mặt (mắt, răng, tóc)

### 7.4. Kiểm tra hiểu biết

- [ ] Grad-CAM hoạt động như thế nào?
- [ ] Tại sao chọn conv layer cuối làm target?
- [ ] Nếu heatmap highlight background thay vì mặt, điều đó có nghĩa gì?

---

## 8. Tuần 13-16: Web Application

### 8.1. Mục tiêu

- [ ] Xây dựng REST API với FastAPI
- [ ] Tạo UI với Gradio
- [ ] Tích hợp ONNX inference
- [ ] Deploy demo (local hoặc Colab)

### 8.2. Tài liệu học

| Tài liệu              | Link                                                                                     | Thời gian | Priority |
| --------------------- | ---------------------------------------------------------------------------------------- | --------- | -------- |
| **FastAPI Tutorial**  | [Official](https://fastapi.tiangolo.com/tutorial/)                                       | 2h        | 🔴       |
| **Gradio Quickstart** | [Official](https://www.gradio.app/guides/quickstart)                                     | 1h        | 🔴       |
| **ONNX Export**       | [PyTorch](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) | 1h        | 🟡       |

### 8.3. Hands-on Exercises

#### Exercise 8.1: Gradio Demo (Nhanh nhất)

```python
# File: app/gradio_demo.py
# Web demo đơn giản với Gradio

import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.efficientnet import create_model
from src.data.dataset import get_val_transforms
from src.xai.gradcam import ExplainabilityEngine

# Load model globally
MODEL_PATH = "outputs/checkpoints/best_model.pt"
model = create_model(checkpoint_path=MODEL_PATH)
model.eval()

transform = get_val_transforms()
explainer = ExplainabilityEngine(model)


def predict_image(image: Image.Image):
    """
    Predict và giải thích một ảnh.

    Args:
        image: PIL Image từ Gradio

    Returns:
        (prediction_text, heatmap_image)
    """
    # Save temp for processing
    temp_path = "/tmp/gradio_input.jpg"
    image.save(temp_path)

    # Visualize
    viz, prob, pred = explainer.visualize(
        image_path=temp_path,
        transform=transform,
    )

    # Format result
    if pred == "FAKE":
        result = f"🚨 ẢNH GIẢ (AI-Generated)\nĐộ tin cậy: {prob:.1%}"
    else:
        result = f"✅ ẢNH THẬT\nĐộ tin cậy: {1-prob:.1%}"

    return result, Image.fromarray(viz)


# Create Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload ảnh chân dung"),
    outputs=[
        gr.Textbox(label="Kết quả"),
        gr.Image(label="Grad-CAM Heatmap"),
    ],
    title="🔍 HolmHz - Phát hiện ảnh AI-Generated",
    description="""
    Upload ảnh chân dung để kiểm tra xem đó là ảnh thật hay ảnh do AI tạo ra.

    **Lưu ý**: Kết quả chỉ mang tính tham khảo, không thay thế kết luận chuyên môn.
    """,
    examples=[
        ["samples/real_example.jpg"],
        ["samples/fake_example.jpg"],
    ],
    theme=gr.themes.Soft(),
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Tạo public link
    )
```

#### Exercise 8.2: FastAPI Backend

```python
# File: app/api.py
# REST API với FastAPI

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import io
from PIL import Image
import base64
import numpy as np

app = FastAPI(
    title="HolmHz API",
    description="API phát hiện ảnh tổng hợp",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class PredictionResponse(BaseModel):
    prediction: str  # "REAL" or "FAKE"
    probability: float
    confidence: float
    heatmap_base64: Optional[str] = None


# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, transform, explainer
    from src.models.efficientnet import create_model
    from src.data.dataset import get_val_transforms
    from src.xai.gradcam import ExplainabilityEngine

    model = create_model(checkpoint_path="outputs/checkpoints/best_model.pt")
    model.eval()
    transform = get_val_transforms()
    explainer = ExplainabilityEngine(model)


@app.get("/")
async def root():
    return {"message": "HolmHz API is running!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    explain: bool = True,
):
    """
    Predict whether an image is real or AI-generated.

    Args:
        file: Image file (JPEG, PNG)
        explain: Include Grad-CAM heatmap

    Returns:
        Prediction with confidence and optional heatmap
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    img_array = np.array(image.resize((224, 224)))
    transformed = transform(image=img_array)
    input_tensor = transformed["image"].unsqueeze(0)

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    prediction = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob

    # Explain
    heatmap_b64 = None
    if explain:
        cam = explainer.explain(input_tensor)
        # Convert to base64
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import cv2

        viz = show_cam_on_image(
            img_array.astype(np.float32) / 255.0,
            cam,
            use_rgb=True,
        )
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        heatmap_b64 = base64.b64encode(buffer).decode()

    return PredictionResponse(
        prediction=prediction,
        probability=prob,
        confidence=confidence,
        heatmap_base64=heatmap_b64,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 8.4. Kiểm tra hiểu biết

- [ ] FastAPI vs Flask: ưu nhược điểm?
- [ ] Gradio phù hợp cho use case nào?
- [ ] Tại sao cần ONNX thay vì PyTorch trực tiếp?

---

## 9. Tài nguyên bổ sung

### 9.1. Papers cần đọc (theo thứ tự ưu tiên)

| #   | Paper                                                                                                | Năm  | Tại sao đọc           |
| --- | ---------------------------------------------------------------------------------------------------- | ---- | --------------------- |
| 1   | [Wang et al. - CNN-generated images are surprisingly easy to spot](https://arxiv.org/abs/1912.11035) | 2020 | Baseline method       |
| 2   | [Frank et al. - Leveraging Frequency Analysis](https://arxiv.org/abs/2003.08685)                     | 2020 | Frequency approach    |
| 3   | [Tan & Le - EfficientNet](https://arxiv.org/abs/1905.11946)                                          | 2019 | Backbone architecture |
| 4   | [Selvaraju et al. - Grad-CAM](https://arxiv.org/abs/1610.02391)                                      | 2017 | XAI method            |

### 9.2. GitHub Repos để tham khảo

| Repo                                                                    | Mô tả                               |
| ----------------------------------------------------------------------- | ----------------------------------- |
| [CNNDetection](https://github.com/PeterWang512/CNNDetection)            | Official implementation Wang et al. |
| [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect) | CLIP-based detection                |
| [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)                 | Benchmark framework                 |
| [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)        | Grad-CAM library                    |

### 9.3. Channels/Blogs để follow

| Resource          | Link                                                | Nội dung           |
| ----------------- | --------------------------------------------------- | ------------------ |
| Yannic Kilcher    | [YouTube](https://www.youtube.com/@YannicKilcher)   | Paper explanations |
| Two Minute Papers | [YouTube](https://www.youtube.com/@TwoMinutePapers) | AI news            |
| PyTorch Blog      | [Official](https://pytorch.org/blog/)               | Tutorials          |
| Weights & Biases  | [Blog](https://wandb.ai/site/articles)              | ML best practices  |

---

## 📅 Learning Schedule Template

```markdown
## Tuần X (DD/MM - DD/MM)

### Mục tiêu tuần này

- [ ] Mục tiêu 1
- [ ] Mục tiêu 2

### Học

- [ ] Resource 1 - Thời gian
- [ ] Resource 2 - Thời gian

### Code

- [ ] Exercise X.X
- [ ] Áp dụng vào HolmHz: ...

### Ghi chú

- Điều học được:
- Câu hỏi còn thắc mắc:
- Kế hoạch tuần sau:
```

---

**Last Updated:** 02/02/2026  
**Author:** AI Assistant for Lê Văn Hoàng
