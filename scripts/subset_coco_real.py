"""
Subset 300 ảnh từ COCO 2017 Val → real_camera_train.

COCO Val chứa 5,000 ảnh camera thật, cực đa dạng (80 categories).
Lấy random 300 ảnh (seed=42) cho reproducibility.

Usage:
    python scripts/subset_coco_real.py

Output:
    data/raw/real/real_camera_train/  (300 images)
"""

import random
import shutil
from pathlib import Path

# === CONFIG ===
SEED = 42
SAMPLE_COUNT = 300
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# COCO extract path (auto-detect subfolder)
COCO_ROOT = Path("data/raw/coco_val2017")
DST_DIR = Path("data/raw/real/real_camera_train")


def find_coco_images(root: Path) -> list[Path]:
    """Tìm tất cả ảnh COCO, tự detect subfolder."""
    # COCO thường extract vào val2017/ subfolder
    candidates = [
        root / "val2017",        # data/raw/coco_val2017/val2017/
        root,                     # data/raw/coco_val2017/ (nếu extract trực tiếp)
    ]

    for folder in candidates:
        if folder.exists():
            images = sorted([
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ])
            if len(images) > 100:
                print(f"  Found images in: {folder}")
                return images

    return []


def main():
    print("=" * 50)
    print("Subset COCO 2017 Val → real_camera_train")
    print("=" * 50)

    # 1. Tìm images
    images = find_coco_images(COCO_ROOT)
    if not images:
        print(f"❌ COCO images not found at: {COCO_ROOT}")
        print(f"   Download từ: http://images.cocodataset.org/zips/val2017.zip")
        print(f"   Extract vào: {COCO_ROOT}/")
        return

    print(f"✅ Found {len(images)} COCO Val images")

    # 2. Random subset
    random.seed(SEED)
    selected = random.sample(images, min(SAMPLE_COUNT, len(images)))
    print(f"📋 Selected {len(selected)} random images (seed={SEED})")

    # 3. Copy
    DST_DIR.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in DST_DIR.iterdir() if f.is_file()}

    copied, skipped = 0, 0
    for img in selected:
        if img.name in existing:
            skipped += 1
            continue
        shutil.copy2(img, DST_DIR / img.name)
        copied += 1

    total_files = len(list(DST_DIR.iterdir()))
    print(f"\n✅ Copied: {copied} new, {skipped} skipped (already exist)")
    print(f"📁 Output: {DST_DIR}/ ({total_files} files)")

    # 4. Verify size diversity
    try:
        from PIL import Image
        sizes = set()
        for img_path in list(DST_DIR.iterdir())[:50]:
            try:
                w, h = Image.open(img_path).size
                sizes.add((w, h))
            except Exception:
                pass
        print(f"📐 Resolution diversity: {len(sizes)} unique sizes in first 50 images")
    except ImportError:
        pass

    print(f"\n📋 Next: python scripts/resize_all.py")


if __name__ == "__main__":
    main()
