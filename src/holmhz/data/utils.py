"""
Data utility functions — factory cho DataLoader.

Cung cấp hàm create_dataloader() để tạo DataLoader từ manifest + config.
Trainer class (Task 1.5) sẽ gọi hàm này.
"""

from pathlib import Path

from torch.utils.data import DataLoader

from .image_dataset import ImageDataset
from .transforms import get_train_transforms, get_val_transforms


def create_dataloader(
    manifest_path: str,
    batch_size: int = 32,
    image_size: int = 224,
    is_training: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Tạo DataLoader từ manifest JSON file.

    Args:
        manifest_path: Đường dẫn tới manifest JSON.
        batch_size: Số ảnh mỗi batch (32 cho train, 64 cho val/test).
        image_size: Kích thước ảnh (224 cho EfficientNet-B0).
        is_training: True → augment + shuffle. False → no augment + no shuffle.
        num_workers: Số thread đọc data song song.
        pin_memory: Pin memory cho GPU transfer.

    Returns:
        DataLoader sẵn sàng sử dụng.

    Example:
        >>> train_loader = create_dataloader("data/manifests/train.json", is_training=True)
        >>> val_loader = create_dataloader("data/manifests/val.json", is_training=False)
    """
    # Chọn transform phù hợp
    if is_training:
        transform = get_train_transforms(image_size)
    else:
        transform = get_val_transforms(image_size)

    # Tạo dataset
    dataset = ImageDataset(
        manifest_path=manifest_path,
        transform=transform,
    )

    # Tạo DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,       # Shuffle chỉ khi training
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_training,     # Drop batch cuối nếu không đủ size (chỉ khi training)
    )

    return loader


def get_dataset_info(manifest_path: str) -> dict:
    """
    Trả về thông tin tổng quan về dataset từ manifest.

    Returns:
        dict với total, real, fake, sources, label_ratio
    """
    import json

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from collections import Counter
    source_counts = Counter(item["source"] for item in data)
    n_real = sum(1 for item in data if item["label"] == 0)
    n_fake = sum(1 for item in data if item["label"] == 1)

    return {
        "total": len(data),
        "real": n_real,
        "fake": n_fake,
        "label_ratio": f"{n_real / len(data):.1%} real / {n_fake / len(data):.1%} fake",
        "sources": dict(source_counts),
    }