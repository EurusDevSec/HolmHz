"""
Subset 3,000 anh Real diverse tu 140k_real_and_fake dataset.

Dataset 140k co 50,000 real images (256x256, diverse objects -- khong chi face).
Ta chi can 3,000 de bo sung training data.

Usage:
    python scripts/subset_140k_real.py

Output:
    data/raw/real/diverse_real/ (3,000 images)
"""

import random
import shutil
from pathlib import Path

# === CONFIG ===
SRC_DIR = Path("data/raw/140k_real_and_fake/real_vs_fake/real-vs-fake/train/real")
DST_DIR = Path("data/raw/real/diverse_real")
SUBSET_SIZE = 3000
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    # Validate source
    if not SRC_DIR.exists():
        print(f"ERROR: Source not found: {SRC_DIR}")
        return

    # Gather all images
    all_images = sorted([
        f for f in SRC_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ])
    print(f"Source: {SRC_DIR}")
    print(f"Total available: {len(all_images)}")

    # Random subset
    random.seed(SEED)
    selected = random.sample(all_images, min(SUBSET_SIZE, len(all_images)))
    print(f"Selected: {len(selected)} images (seed={SEED})")

    # Copy to destination
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing (resume support)
    existing = set(f.name for f in DST_DIR.iterdir() if f.is_file())
    copied = 0
    skipped = 0

    for img_path in selected:
        if img_path.name in existing:
            skipped += 1
            continue
        shutil.copy2(img_path, DST_DIR / img_path.name)
        copied += 1

    total = copied + skipped
    print(f"\nDone! {total} images in {DST_DIR}")
    print(f"  Copied: {copied}, Skipped (existing): {skipped}")
    print(f"\nNext: python scripts/split_real_pexels.py")


if __name__ == "__main__":
    main()