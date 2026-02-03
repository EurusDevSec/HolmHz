# 🐍 Python OOP Best Practices cho AI Engineer

Tài liệu này tổng hợp các kiến thức cốt lõi về Object-Oriented Programming (OOP) trong Python, tập trung vào các pattern thường dùng trong Deep Learning (PyTorch).

---

## 1. Class & Type Hinting (Best Practice)

Trong các dự án AI lớn như HolmHz, việc viết code rõ ràng quan trọng hơn viết code ngắn. Luôn sử dụng **Type Hints**.

### ❌ Bad Code

```python
class ImageLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        # Không biết trả về gì, path là string hay Path object?
        pass
```

### ✅ Good Code (HolmHz Style)

```python
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np

class ImageLoader:
    """Quản lý việc load ảnh từ thư mục."""

    def __init__(self, root_dir: str):
        self.root_dir: Path = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

    def load_image(self, filename: str) -> np.ndarray:
        """Load ảnh và convert sang numpy array."""
        file_path = self.root_dir / filename
        with Image.open(file_path) as img:
            return np.array(img.convert('RGB'))
```

---

## 2. Magic Methods (Dunder Methods)

Trong PyTorch, các "Magic Methods" (bắt đầu và kết thúc bằng `__`) cực kỳ quan trọng để custom **Dataset** và **Transforms**.

### 2.1. `__len__` và `__getitem__` (Cốt lõi của Dataset)

Để Pytorch `DataLoader` hoạt động, object của bạn phải cư xử như một list.

```python
class MyDataset:
    def __init__(self, data: List[str]):
        self.data = data

    def __len__(self) -> int:
        """Trả về tổng số sample."""
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        """Lấy 1 sample tại index cụ thể."""
        return self.data[idx]

# Sử dụng
ds = MyDataset(["img1.jpg", "img2.jpg", "img3.jpg"])
print(len(ds))      # Gọi __len__: 3
print(ds[1])        # Gọi __getitem__: img2.jpg
```

### 2.2. `__call__` (Biến object thành function)

Rất phổ biến trong các class **Data Preprocessing/Data Augmentation**.

```python
class AddGaussianNoise:
    def __init__(self, mean: float = 0., std: float = 1.):
        self.mean = mean
        self.std = std

    def __call__(self, img_array: np.ndarray) -> np.ndarray:
        """Logic chạy khi gọi object()"""
        noise = np.random.normal(self.mean, self.std, img_array.shape)
        return img_array + noise

# Sử dụng
augmentor = AddGaussianNoise(std=0.1)
# Gọi object như một hàm:
noisy_img = augmentor(original_img)
```

### 2.3. `__repr__` (Debug dễ hơn)

Giúp in ra thông tin class dễ đọc khi print.

```python
class ModelConfig:
    def __init__(self, lr: float, batch_size: int):
        self.lr = lr
        self.batch_size = batch_size

    def __repr__(self) -> str:
        return f"Config(lr={self.lr}, bs={self.batch_size})"

conf = ModelConfig(0.001, 32)
print(conf) # Config(lr=0.001, bs=32) thay vì <__main__.ModelConfig object at 0x...>
```

---

## 3. Inheritance (Kế thừa)

Trong Deep Learning, bạn chủ yếu kế thừa từ `torch.nn.Module` (để build model) hoặc `torch.utils.data.Dataset`.

### Ví dụ: Build một Block trong Model

```python
import torch.nn as nn

# Base class là nn.Module
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        # Quan trọng: Luôn gọi super().__init__() đầu tiên
        super().__init__()

        # Composition: Class này chứa các class khác
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))
```

---

## 📝 Hands-on Challenge cho bạn

Hãy thử viết một class `HolmHzDataset` trong file script tạm để thực hành:

1.  Kế thừa (giả lập) từ `object`.
2.  Nhận vào list các đường dẫn ảnh.
3.  Implement `__len__` trả về số lượng ảnh.
4.  Implement `__getitem__` trả về dict `{"image_path": ..., "label": 0}` (giả sử label luôn là 0).
5.  Implement `__call__` để khi gọi `dataset()` thì nó in ra "Dataset for Deepfake Detection".

```python
# CODE CỦA BẠN Ở ĐÂY:
class HolmHzDataset:
    pass
```
