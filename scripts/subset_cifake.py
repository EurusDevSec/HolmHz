# scripts/subset_cifake.py — Chọn random subset từ CIFAKE
"""
Lấy random subset từ CIFAKE dataset.
CIFAKE có 60K real + 60K fake, nhưng ta chỉ cần 7K mỗi loại cho training.
- 5K train + 1K val + 1K test = 7K ảnh mỗi loại

Cấu trúc gốc CIFAKE (giữ nguyên, KHÔNG flatten):
  data/raw/cifake/train/REAL/  (50K, filenames: 0.png → 49999.png)
  data/raw/cifake/train/FAKE/  (50K)
  data/raw/cifake/test/REAL/   (10K, filenames: 0.png → 9999.png)
  data/raw/cifake/test/FAKE/   (10K)

⚠️ Filename TRÙNG giữa train/ và test/ → thêm prefix khi copy sang subset.
"""
import shutil
import random
from pathlib import Path

random.seed(42)  # Seed cố định để reproducible

CIFAKE_ROOT = Path("data/raw/cifake")


def collect_cifake_images(label: str) -> list[Path]:
    """
    Thu thập tất cả ảnh CIFAKE cho 1 label (REAL hoặc FAKE).
    Gộp cả train/ và test/ vào 1 list.
    """
    train_dir = CIFAKE_ROOT / "train" / label
    test_dir = CIFAKE_ROOT / "test" / label

    images = []
    for d in [train_dir, test_dir]:
        if d.exists():
            images.extend(sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")))

    print(f"  Found {len(images)} {label} images (train + test)")
    return images


def subset_to_folder(images: list[Path], dst_dir: str, count: int) -> int:
    """
    Lấy random `count` ảnh, copy sang dst_dir.
    Thêm prefix (train_ hoặc test_) để tránh trùng filename.
    """
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    selected = random.sample(images, min(count, len(images)))

    for img_path in selected:
        # Prefix = tên folder cha-cha (train hoặc test)
        split_name = img_path.parent.parent.name  # "train" hoặc "test"
        new_name = f"{split_name}_{img_path.name}"
        shutil.copy2(img_path, dst / new_name)

    print(f"  Copied {len(selected)} images → {dst}")
    return len(selected)


if __name__ == "__main__":
    print("📦 Collecting CIFAKE Real images...")
    real_images = collect_cifake_images("REAL")
    subset_to_folder(real_images, "data/raw/real/cifake_subset", 7000)

    print("\n📦 Collecting CIFAKE Fake images...")
    fake_images = collect_cifake_images("FAKE")
    subset_to_folder(fake_images, "data/raw/fake_diffusion/cifake_subset", 7000)

    print("\n✅ Subset done! Dùng folder *_subset cho pipeline.")
    print("Nếu muốn tăng data sau → chạy lại với count lớn hơn.")