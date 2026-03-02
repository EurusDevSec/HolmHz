"""
Data utility functions — factory cho DataLoader.

Cung cấp hàm create_dataloader() để tạo DataLoader từ manifest + config.
Trainer class (Task 1.5) sẽ gọi hàm này.
"""

import json
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .image_dataset import ImageDataset
from .transforms import get_train_transforms, get_val_transforms


def compute_source_weights(manifest_path: str) -> list[float]:
    """
    Tính sample weights dựa trên source để dùng với WeightedRandomSampler.

    Mỗi source sẽ có xác suất được sample bằng nhau trong 1 epoch,
    bất kể số lượng ảnh gốc. Giúp:
    - Downsample cifake (9800 → ~3000 effective)
    - Upsample tristanzhang_train (140 → ~3000 effective)
    - Balance tất cả sources

    Args:
        manifest_path: Đường dẫn tới manifest JSON.

    Returns:
        list[float] — weight cho từng sample (cùng thứ tự với manifest).
    """
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    # Đếm số ảnh mỗi source
    source_counts = Counter(item["source"] for item in data)
    max_count = max(source_counts.values())

    # Weight = max_count / source_count
    # → source ít ảnh → weight cao → được sample nhiều hơn
    source_weights = {src: max_count / count for src, count in source_counts.items()}

    # Gán weight cho từng sample
    sample_weights = [source_weights[item["source"]] for item in data]

    return sample_weights


def create_weighted_sampler(manifest_path: str, num_samples: int | None = None) -> WeightedRandomSampler:
    """
    Tạo WeightedRandomSampler để cân bằng sources trong training.

    Args:
        manifest_path: Đường dẫn tới manifest JSON.
        num_samples: Số samples mỗi epoch. None = len(dataset).

    Returns:
        WeightedRandomSampler instance.
    """
    sample_weights = compute_source_weights(manifest_path)

    if num_samples is None:
        num_samples = len(sample_weights)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=num_samples,
        replacement=True,  # Cần replacement vì minority sources < num_samples
    )

    return sampler


def create_dataloader(
    manifest_path: str,
    batch_size: int = 32,
    image_size: int = 224,
    is_training: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
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
        use_weighted_sampler: True → dùng WeightedRandomSampler thay shuffle.
            Cân bằng sources (downsample cifake, upsample minority).

    Returns:
        DataLoader sẵn sàng sử dụng.

    Example:
        >>> train_loader = create_dataloader("data/manifests/train.json", is_training=True)
        >>> val_loader = create_dataloader("data/manifests/val.json", is_training=False)
        >>> # Với balanced sampling:
        >>> train_loader = create_dataloader("data/manifests/train.json", is_training=True, use_weighted_sampler=True)
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

    # Weighted sampler (chỉ khi training + được bật)
    sampler = None
    shuffle = is_training
    if is_training and use_weighted_sampler:
        sampler = create_weighted_sampler(manifest_path)
        shuffle = False  # Không dùng shuffle khi có sampler

    # Tạo DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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

    with open(manifest_path, encoding="utf-8") as f:
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
