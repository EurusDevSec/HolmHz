"""
Split real_pexels: 300 anh cho training, 200 giu lai cho OOD test.

Hien tai: 500 anh real_pexels deu nam trong OOD test set.
Ta can chuyen 300 anh vao training de model hoc "anh phong canh = Real".
Giu lai 200 anh de test OOD (van co tinh generalization).

Usage:
    python scripts/split_real_pexels.py

Output:
    data/raw/real/real_pexels_train/  (300 images)
    data/raw/ood_test/real_pexels/    (giu nguyen 500 -- manifest se chi dung 200)
"""

import random
import shutil
from pathlib import Path

# === CONFIG ===
SRC_DIR = Path("data/raw/ood_test/real_pexels")
DST_DIR = Path("data/raw/real/real_pexels_train")
TRAIN_COUNT = 300
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    if not SRC_DIR.exists():
        print(f"ERROR: Source not found: {SRC_DIR}")
        return

    # Gather all images
    all_images = sorted([
        f for f in SRC_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ])
    print(f"Source: {SRC_DIR}")
    print(f"Total: {len(all_images)}")

    # Random split: 300 train, 200 test
    random.seed(SEED)
    random.shuffle(all_images)  # in-place shuffle for reproducibility

    # Re-sort after shuffle for consistency
    shuffled = list(all_images)  # already shuffled
    train_images = shuffled[:TRAIN_COUNT]
    test_images = shuffled[TRAIN_COUNT:]

    print(f"Train: {len(train_images)}")
    print(f"Test:  {len(test_images)}")

    # Copy train images (DO NOT move -- keep originals in ood_test)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    existing = set(f.name for f in DST_DIR.iterdir() if f.is_file())
    copied = 0
    skipped = 0

    for img_path in train_images:
        if img_path.name in existing:
            skipped += 1
            continue
        shutil.copy2(img_path, DST_DIR / img_path.name)
        copied += 1

    print(f"\nCopied to {DST_DIR}: {copied} new, {skipped} skipped")

    # Save the test-only file list (for build_splits.py to use)
    test_names = sorted([f.name for f in test_images])
    list_path = Path("data/manifests/real_pexels_test_only.txt")
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, "w") as f:
        for name in test_names:
            f.write(name + "\n")

    print(f"Test-only list saved: {list_path} ({len(test_names)} files)")
    print(f"\nNext: python scripts/resize_all.py  (to resize new data)")


if __name__ == "__main__":
    main()