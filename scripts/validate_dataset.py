"""
scripts/validate_dataset.py

Kiểm tra data integrity cho data/processed/:
1. Ảnh không bị corrupt (mở được bằng PIL)
2. Tất cả ảnh đều là 224×224
3. Không có file 0 bytes
4. Không có folder rỗng
"""
import sys
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
EXPECTED_SIZE = (224, 224)


def validate_folder(folder: Path) -> dict:
    """Kiểm tra toàn bộ ảnh trong folder."""
    results = {
        "total": 0,
        "valid": 0,
        "corrupt": [],
        "wrong_size": [],
        "zero_bytes": [],
    }

    images = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ])
    results["total"] = len(images)

    for img_path in tqdm(images, desc=f"  {folder.name}", leave=False, unit="img"):
        # Check 0 bytes
        if img_path.stat().st_size == 0:
            results["zero_bytes"].append(str(img_path.name))
            continue

        try:
            img = Image.open(img_path)
            img.verify()  # Verify file is not corrupt

            # Re-open (verify() closes the file)
            img = Image.open(img_path)

            if img.size != EXPECTED_SIZE:
                results["wrong_size"].append(f"{img_path.name}: {img.size}")
            else:
                results["valid"] += 1

        except Exception as e:
            results["corrupt"].append(f"{img_path.name}: {e}")

    return results


def main():
    processed = Path("data/processed")

    folders = [
        # Training data
        processed / "train/real/cifake",
        processed / "train/real/ffhq",
        processed / "train/fake_gan/stylegan",
        processed / "train/fake_diffusion/cifake",
        processed / "train/fake_diffusion/sd15",
        # OOD test data
        processed / "ood_test/tristanzhang_fake",
        processed / "ood_test/real_pexels",
        processed / "ood_test/flux",
        processed / "ood_test/real_camera",
    ]

    all_ok = True
    total_valid = 0
    total_checked = 0

    print("=" * 55)
    print("  DATA VALIDATION — HolmHz")
    print("=" * 55)

    for folder in folders:
        if not folder.exists():
            rel = folder.relative_to(processed)
            print(f"\n  [SKIP] {rel} — not found")
            continue

        results = validate_folder(folder)
        rel = folder.relative_to(processed)
        total_valid += results["valid"]
        total_checked += results["total"]

        if results["valid"] == results["total"] and results["total"] > 0:
            print(f"\n  [OK]   {rel}: {results['valid']}/{results['total']} valid")
        elif results["total"] == 0:
            print(f"\n  [EMPTY] {rel}: no images")
            all_ok = False
        else:
            print(f"\n  [WARN] {rel}: {results['valid']}/{results['total']} valid")
            all_ok = False

            if results["corrupt"]:
                for item in results["corrupt"][:3]:
                    print(f"           corrupt: {item}")
                if len(results["corrupt"]) > 3:
                    print(f"           ... and {len(results['corrupt']) - 3} more")

            if results["wrong_size"]:
                for item in results["wrong_size"][:3]:
                    print(f"           wrong size: {item}")
                if len(results["wrong_size"]) > 3:
                    print(f"           ... and {len(results['wrong_size']) - 3} more")

            if results["zero_bytes"]:
                for item in results["zero_bytes"][:3]:
                    print(f"           zero bytes: {item}")
                if len(results["zero_bytes"]) > 3:
                    print(f"           ... and {len(results['zero_bytes']) - 3} more")

    print(f"\n{'=' * 55}")
    print(f"  Total: {total_valid:,}/{total_checked:,} images valid ({EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]})")
    if all_ok:
        print(f"  >>> ALL DATA VALID <<<")
    else:
        print(f"  >>> ISSUES FOUND — fix before proceeding <<<")
    print("=" * 55)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
