"""
HolmHz Data Reset — Download verified datasets from Kaggle.

Downloads:
1. DeepDetect-2025: 100K+ images (people, animals, nature, urban)
   - Real: ~60K images
   - Fake: ~55K from StyleGAN3/DALLE-3/Midjourney/SD3
2. GenImage Subset: 12K images (SD/MJ/BigGAN)

Prerequisites:
   pip install kaggle
   Place kaggle.json in C:/Users/<USER>/.kaggle/kaggle.json
   Get API key: https://www.kaggle.com/settings -> API -> Create New Token

Usage:
   python scripts/download_datasets.py
"""

import os
import sys
import shutil
import json
import random
from pathlib import Path

# Kaggle dataset slugs
DATASETS = {
    "deepdetect2025": "aldoganzozo/deepdetect-2025",
    "genimage_subset": "yangsangtai/genimage-subset-detection",
}

DATA_ROOT = Path("data/raw_v2")


def download_kaggle(slug: str, dest: str):
    """Download a Kaggle dataset using the CLI."""
    os.makedirs(dest, exist_ok=True)
    cmd = f'kaggle datasets download -d {slug} -p "{dest}" --unzip'
    print(f"  Downloading: {slug}")
    print(f"  Command: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"  ERROR: Download failed (exit code {ret})")
        print("  Make sure kaggle.json is in ~/.kaggle/")
        return False
    return True


def scan_images(folder: Path, extensions=None):
    """Recursively find all images."""
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = []
    for f in folder.rglob("*"):
        if f.suffix.lower() in extensions and f.is_file():
            images.append(f)
    return sorted(images)


def build_manifest(real_dirs: list, fake_dirs: list, output_dir: Path,
                   max_per_source: int = 5000, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """Build train/val/test manifests from real and fake directories."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Collect real images
    for label, dirs in [(0, real_dirs), (1, fake_dirs)]:
        for source_name, source_dir in dirs:
            images = scan_images(Path(source_dir))
            if len(images) > max_per_source:
                random.seed(42)
                images = random.sample(images, max_per_source)
            for img in images:
                all_samples.append({
                    "path": str(img),
                    "label": label,
                    "source": source_name,
                })
            label_name = "FAKE" if label == 1 else "REAL"
            print(f"  [{label_name}] {source_name}: {len(images)} images")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(all_samples)

    # Split by source (each source goes entirely into one split)
    sources = {}
    for s in all_samples:
        src = s["source"]
        if src not in sources:
            sources[src] = []
        sources[src].append(s)

    # Simple split: 80% train, 10% val, 10% test (by sample, stratified)
    train, val, test = [], [], []
    for src, items in sources.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])

    for name, data in [("train.json", train), ("val.json", val), ("test.json", test)]:
        path = output_dir / name
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        n_fake = sum(1 for d in data if d["label"] == 1)
        n_real = len(data) - n_fake
        print(f"  {name}: {len(data)} samples (real={n_real}, fake={n_fake})")

    return train, val, test


def main():
    print("=" * 60)
    print("HolmHz Data Reset — Download Verified Datasets")
    print("=" * 60)

    # Step 1: Download from Kaggle
    for name, slug in DATASETS.items():
        dest = DATA_ROOT / name
        if dest.exists() and any(dest.rglob("*")):
            print(f"\n[SKIP] {name} already exists at {dest}")
        else:
            print(f"\n[DOWNLOAD] {name}")
            success = download_kaggle(slug, str(dest))
            if not success:
                print(f"  Skipping {name}")

    # Step 2: Scan downloaded data
    print("\n" + "=" * 60)
    print("Step 2: Scanning downloaded data")
    print("=" * 60)

    for name in DATASETS:
        dest = DATA_ROOT / name
        if dest.exists():
            images = scan_images(dest)
            print(f"  {name}: {len(images)} images found")
            # Show directory structure
            for d in sorted(set(f.parent for f in images))[:20]:
                count = sum(1 for f in images if f.parent == d)
                print(f"    {d.relative_to(dest)}: {count} files")
        else:
            print(f"  {name}: NOT DOWNLOADED")

    print("\n" + "=" * 60)
    print("Step 3: After download, run build_manifest() to create splits")
    print("=" * 60)
    print("See scripts/prepare_data_v2.py for manifest building")


if __name__ == "__main__":
    main()
