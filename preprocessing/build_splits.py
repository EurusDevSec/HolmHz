"""
Tạo manifest JSON files cho train/val/test split.

Input:  data/processed/train/{real,fake_gan,fake_diffusion}/{source}/
        data/processed/ood_test/{source}/
Output: data/manifests/train.json
        data/manifests/val.json
        data/manifests/test_id.json
        data/manifests/test_ood.json

Logic:
  1. Scan tất cả ảnh trong data/processed/train/
  2. Gán label: real/ → 0, fake_*/ → 1
  3. Chia stratified 70/15/15 (seed=42, reproducible)
  4. OOD test: tách riêng, không chia
  5. Lưu 4 file JSON manifest

Usage:
  python preprocessing/build_splits.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# === CẤU HÌNH ===
PROCESSED_DIR = Path("data/processed")
MANIFESTS_DIR = Path("data/manifests")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# === LABEL MAPPING ===
# Folder name → label
CATEGORY_LABELS = {
    "real": 0,        # Ảnh thật
    "fake_gan": 1,    # Ảnh GAN (StyleGAN)
    "fake_diffusion": 1,  # Ảnh Diffusion (CIFAKE, SD v1.5)
}


def scan_folder(base_dir: Path) -> list[dict]:
    """
    Scan 1 category folder, trả về list các entry.

    Ví dụ: scan_folder("data/processed/train/real")
    → [{"path": "data/processed/.../00001.png", "label": 0, "source": "cifake", "category": "real"}, ...]
    """
    entries = []

    if not base_dir.exists():
        print(f"  ⏭️  Bỏ qua (không tồn tại): {base_dir}")
        return entries

    # Lặp qua các sub-folder (mỗi sub-folder = 1 source)
    for source_dir in sorted(base_dir.iterdir()):
        if not source_dir.is_dir():
            continue

        source_name = source_dir.name
        category_name = base_dir.name  # "real", "fake_gan", "fake_diffusion"
        label = CATEGORY_LABELS.get(category_name, -1)

        if label == -1:
            print(f"  ⚠️  Không biết label cho category: {category_name}")
            continue

        # Scan ảnh
        images = sorted([
            f for f in source_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        for img_path in images:
            entries.append({
                "path": str(img_path.as_posix()),  # Forward slash cho JSON
                "label": label,
                "source": source_name,
                "category": category_name,
            })

        print(f"  ✅ {category_name}/{source_name}: {len(images)} ảnh (label={label})")

    return entries


def scan_ood_folder(ood_dir: Path) -> list[dict]:
    """
    Scan OOD test folder riêng.
    OOD label: fake folders → 1, real folders → 0
    """
    entries = []

    if not ood_dir.exists():
        print(f"  ⏭️  OOD folder không tồn tại: {ood_dir}")
        return entries

    # Mapping OOD sources → label
    OOD_LABELS = {
        "tristanzhang_fake": 1,  # Mixed SD+MJ+DALLE
        "flux": 1,               # FLUX.1-schnell
        "real_pexels": 0,        # Real photos (Pexels/Unsplash)
        "real_camera": 0,        # Real camera photos (Unsplash API)
    }

    for source_dir in sorted(ood_dir.iterdir()):
        if not source_dir.is_dir():
            continue

        source_name = source_dir.name
        label = OOD_LABELS.get(source_name, -1)

        if label == -1:
            print(f"  ⚠️  Không biết label OOD cho: {source_name}")
            continue

        images = sorted([
            f for f in source_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        for img_path in images:
            entries.append({
                "path": str(img_path.as_posix()),
                "label": label,
                "source": source_name,
                "category": "ood",
            })

        label_str = "fake" if label == 1 else "real"
        print(f"  ✅ ood_test/{source_name}: {len(images)} ảnh (label={label} → {label_str})")

    return entries


def stratified_split(entries: list[dict], train_r: float, val_r: float, seed: int):
    """
    Chia stratified theo source: mỗi source được chia đúng tỷ lệ train/val/test.

    Tại sao stratified theo source?
    → Đảm bảo mỗi nguồn (cifake, ffhq, stylegan, sd15) xuất hiện
      đúng tỷ lệ trong cả 3 tập. Nếu chia random, có thể train không có
      sd15 mà test toàn sd15 → kết quả sai.
    """
    random.seed(seed)

    # Nhóm theo source
    by_source = defaultdict(list)
    for entry in entries:
        by_source[entry["source"]].append(entry)

    train_data, val_data, test_data = [], [], []

    for source, items in sorted(by_source.items()):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        # test = phần còn lại

        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])

    # Shuffle lại sau khi chia (để train không bị nhóm theo source)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def save_manifest(data: list[dict], filepath: Path):
    """Lưu manifest ra JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved: {filepath} ({len(data)} entries)")


def main():
    print("=" * 60)
    print("BUILD SPLITS — HolmHz Data Pipeline")
    print("=" * 60)

    # === 1. Scan tất cả ảnh trong train/ ===
    print("\n📂 Scanning training data...")
    train_dir = PROCESSED_DIR / "train"

    all_entries = []
    for category in ["real", "fake_gan", "fake_diffusion"]:
        entries = scan_folder(train_dir / category)
        all_entries.extend(entries)

    print(f"\n📊 Tổng cộng training data: {len(all_entries)} ảnh")
    print(f"   Real: {sum(1 for e in all_entries if e['label'] == 0)}")
    print(f"   Fake: {sum(1 for e in all_entries if e['label'] == 1)}")

    # === 2. Chia stratified train/val/test ===
    print(f"\n✂️  Chia stratified ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}, seed={SEED})...")
    train_data, val_data, test_data = stratified_split(
        all_entries, TRAIN_RATIO, VAL_RATIO, SEED
    )

    # === 3. Scan OOD test ===
    print("\n📂 Scanning OOD test data...")
    ood_data = scan_ood_folder(PROCESSED_DIR / "ood_test")

    # === 4. Thống kê ===
    print(f"\n{'=' * 60}")
    print("📊 SPLIT RESULTS:")
    print(f"{'=' * 60}")

    for name, data in [("Train", train_data), ("Val", val_data),
                        ("Test ID", test_data), ("Test OOD", ood_data)]:
        n_real = sum(1 for e in data if e["label"] == 0)
        n_fake = sum(1 for e in data if e["label"] == 1)
        sources = set(e["source"] for e in data)
        print(f"  {name:10s}: {len(data):6d} ảnh ({n_real} real, {n_fake} fake) — sources: {sorted(sources)}")

    # === 5. Lưu manifests ===
    print(f"\n💾 Saving manifests...")
    save_manifest(train_data, MANIFESTS_DIR / "train.json")
    save_manifest(val_data, MANIFESTS_DIR / "val.json")
    save_manifest(test_data, MANIFESTS_DIR / "test_id.json")
    save_manifest(ood_data, MANIFESTS_DIR / "test_ood.json")

    # === 6. Verify ===
    total = len(train_data) + len(val_data) + len(test_data)
    assert total == len(all_entries), f"Split mismatch: {total} != {len(all_entries)}"
    print(f"\n✅ DONE! {total} train/val/test + {len(ood_data)} OOD = {total + len(ood_data)} tổng cộng")
    print(f"   Manifests saved to: {MANIFESTS_DIR}/")


if __name__ == "__main__":
    main()