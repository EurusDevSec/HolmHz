# scripts/subset_stylegan.py
"""
Lấy random 5K ảnh StyleGAN fake từ 140k-real-and-fake-faces.
Chỉ lấy từ train/fake/ — đủ lớn (50K), không cần merge splits.

Dataset structure:
  data/raw/140k_real_and_fake/real_vs_fake/real-vs-fake/
    ├── train/fake/   (50,000 ảnh StyleGAN) ← LẤY TỪ ĐÂY
    ├── train/real/   (50,000 ảnh thật)     ← Dùng thay FFHQ nếu chưa có
    ├── test/fake/    (10,000 ảnh)
    ├── test/real/    (10,000 ảnh)
    ├── valid/fake/   (10,000 ảnh)
    └── valid/real/   (10,000 ảnh)

Tại sao chỉ lấy train/fake/?
  - train/fake/ đã có 50K ảnh, dư để lấy 3-5K
  - HolmHz tự chia train/val/test ở Task 1.3 → split gốc không quan trọng
  - Không cần merge test/valid vì sẽ gây phức tạp không cần thiết
"""
import shutil
import random
from pathlib import Path

random.seed(42)

BASE = Path("data/raw/140k_real_and_fake/real_vs_fake/real-vs-fake")


def subset_folder(src: Path, dst: Path, count: int) -> int:
    """Lấy random `count` ảnh từ src, copy sang dst."""
    if not src.exists():
        print(f"  ❌ Không tìm thấy: {src}")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    images = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")))

    if not images:
        print(f"  ❌ Không có ảnh trong: {src}")
        return 0

    print(f"  Found {len(images)} images in {src}")
    selected = random.sample(images, min(count, len(images)))
    for img in selected:
        shutil.copy2(img, dst / img.name)

    print(f"  Copied {len(selected)} → {dst}")
    return len(selected)


if __name__ == "__main__":
    # GAN Fake → data/raw/fake_gan/stylegan/
    print("📦 StyleGAN fake faces (từ train/fake/)...")
    n_fake = subset_folder(
        src=BASE / "train" / "fake",
        dst=Path("data/raw/fake_gan/stylegan"),
        count=5000,
    )

    # Real faces → data/raw/real/ffhq/ (thay thế FFHQ nếu chưa có)
    ffhq_dir = Path("data/raw/real/ffhq")
    existing_ffhq = len(list(ffhq_dir.glob("*.jpg")) + list(ffhq_dir.glob("*.png"))) if ffhq_dir.exists() else 0

    if existing_ffhq == 0:
        print("\n📦 Real faces (thay thế FFHQ — từ train/real/)...")
        n_real = subset_folder(
            src=BASE / "train" / "real",
            dst=ffhq_dir,
            count=5000,
        )
    else:
        print(f"\n⏭️  Bỏ qua real faces — {ffhq_dir} đã có {existing_ffhq} ảnh")
        n_real = existing_ffhq

    print(f"\n{'='*50}")
    print(f"✅ Done!")
    print(f"   fake_gan/stylegan: {n_fake} ảnh")
    print(f"   real/ffhq:         {n_real} ảnh")
    print("\nBước tiếp: chạy scripts/resize_all.py")
