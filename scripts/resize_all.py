"""
scripts/resize_all.py

Resize tất cả ảnh trong data/raw/ về 224×224 → save vào data/processed/.
Output format: PNG (lossless, consistent across all sources).

Cấu trúc folder:
  data/raw/real/cifake_subset/          → data/processed/train/real/cifake/
  data/raw/real/ffhq/                   → data/processed/train/real/ffhq/
  data/raw/fake_gan/stylegan/           → data/processed/train/fake_gan/stylegan/
  data/raw/fake_diffusion/cifake_subset → data/processed/train/fake_diffusion/cifake/
  data/raw/fake_diffusion/sd15/         → data/processed/train/fake_diffusion/sd15/
  data/raw/ood_test/*/                  → data/processed/test_ood/*/

Usage:
  python scripts/resize_all.py
"""

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import sys
import os
import time

# Fix Windows console encoding for Unicode/emoji
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

TARGET_SIZE = (224, 224)
RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")

# Increase PIL limit for large images (real_pexels has ~4480x6272 images)
Image.MAX_IMAGE_PIXELS = 300_000_000

# Image extensions to process
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resize_folder(src_dir: Path, dst_dir: Path, save_format: str = "PNG") -> dict:
    """
    Resize tất cả ảnh trong folder về TARGET_SIZE.

    Args:
        src_dir: Đường dẫn folder chứa ảnh gốc
        dst_dir: Đường dẫn folder output
        save_format: Format ảnh output (PNG/JPEG)

    Returns:
        dict với success, errors, skipped counts
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Gather all image files
    images = sorted([
        f for f in src_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ])

    if not images:
        print(f"  ⏭️  No images found in {src_dir}")
        return {"success": 0, "errors": 0, "skipped": 0}

    # Check already processed (for resume support)
    existing = set(f.stem for f in dst_dir.iterdir() if f.is_file())

    success = 0
    errors = 0
    skipped = 0

    ext = ".png" if save_format == "PNG" else ".jpg"

    for img_path in tqdm(images, desc=f"  {src_dir.name}", unit="img"):
        # Skip if already processed
        if img_path.stem in existing:
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(dst_dir / f"{img_path.stem}{ext}")
            success += 1
        except Exception as e:
            errors += 1
            if errors <= 5:  # Chỉ print 5 lỗi đầu
                print(f"  ⚠️  Error: {img_path.name} — {e}")

    print(f"  ✅ {success} resized, {skipped} skipped (already done), {errors} errors")
    return {"success": success, "errors": errors, "skipped": skipped}


def main():
    start_time = time.time()

    # ──────────────────────────────────────────────────────────────
    # Mapping: (raw_relative_path, processed_relative_path)
    #
    # Training data → data/processed/train/
    # OOD test data → data/processed/test_ood/
    # ──────────────────────────────────────────────────────────────
    folders_to_process = [
        # ── Training: Real ──
        ("real/cifake_subset",          "train/real/cifake"),
        ("real/ffhq",                   "train/real/ffhq"),
        # ── Training: Fake GAN ──
        ("fake_gan/stylegan",           "train/fake_gan/stylegan"),
        # ── Training: Fake Diffusion ──
        ("fake_diffusion/cifake_subset","train/fake_diffusion/cifake"),
        ("fake_diffusion/sd15",         "train/fake_diffusion/sd15"),
        # ── OOD Test: Fake ──
        ("ood_test/tristanzhang_fake",  "test_ood/tristanzhang_fake"),
        ("ood_test/flux",               "test_ood/flux"),
        # ── OOD Test: Real ──
        ("ood_test/real_pexels",        "test_ood/real_pexels"),
        ("ood_test/real_camera",        "test_ood/real_camera"),
    ]

    stats = {}
    total = 0

    print("=" * 60)
    print(f"🔄 Resize all images to {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"   Source: {RAW_ROOT.resolve()}")
    print(f"   Dest:   {PROCESSED_ROOT.resolve()}")
    print("=" * 60)

    for src_rel, dst_rel in folders_to_process:
        src = RAW_ROOT / src_rel
        dst = PROCESSED_ROOT / dst_rel

        if not src.exists():
            print(f"\n⏭️  Skip (not found): {src}")
            continue

        # Count source files
        n_src = len([f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
        print(f"\n📁 {src_rel} ({n_src} images) → {dst_rel}")

        result = resize_folder(src, dst)
        count = result["success"] + result["skipped"]
        stats[dst_rel] = {
            "count": count,
            "source": str(src_rel),
            "new_resized": result["success"],
            "skipped_existing": result["skipped"],
            "errors": result["errors"],
        }
        total += count

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    # Build summary grouped by role
    summary = {
        "target_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
        "total_images": total,
        "train": {},
        "test_ood": {},
    }

    for dst_rel, info in stats.items():
        parts = dst_rel.split("/", 1)
        group = parts[0]  # "train" or "test_ood"
        name = parts[1] if len(parts) > 1 else dst_rel

        if group == "train":
            summary["train"][name] = info["count"]
        elif group == "test_ood":
            summary["test_ood"][name] = info["count"]

    summary["train"]["_total"] = sum(
        v for k, v in summary["train"].items() if k != "_total"
    )
    summary["test_ood"]["_total"] = sum(
        v for k, v in summary["test_ood"].items() if k != "_total"
    )

    # Save stats
    stats_path = Path("data/manifests/dataset_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"📊 TỔNG: {total} ảnh đã resize về {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"⏱️  Thời gian: {elapsed:.1f}s")
    print(f"📄 Stats saved: {stats_path}")
    print(f"\n📋 Chi tiết:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
