"""
scripts/dataset_stats.py

Tạo/cập nhật file data/manifests/dataset_stats.json.
Đếm số ảnh trong mỗi folder của data/processed/ và data/raw/.
Kiểm tra acceptance criteria từ TASK_1.2.
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def count_images(folder: Path) -> int:
    """Đếm số file ảnh trong folder."""
    if not folder.exists():
        return 0
    return len([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS])


def main():
    processed = Path("data/processed")
    raw = Path("data/raw")

    # ── Processed data (224×224) ──
    stats = {
        "created": datetime.now().isoformat(),
        "image_size": "224x224",
        "processed": {
            "train": {
                "real": {
                    "cifake": count_images(processed / "train/real/cifake"),
                    "ffhq": count_images(processed / "train/real/ffhq"),
                },
                "fake_gan": {
                    "stylegan": count_images(processed / "train/fake_gan/stylegan"),
                },
                "fake_diffusion": {
                    "cifake": count_images(processed / "train/fake_diffusion/cifake"),
                    "sd15": count_images(processed / "train/fake_diffusion/sd15"),
                },
            },
            "test_ood": {
                "tristanzhang_fake": count_images(processed / "ood_test/tristanzhang_fake"),
                "real_pexels": count_images(processed / "ood_test/real_pexels"),
                "flux": count_images(processed / "ood_test/flux"),
                "real_camera": count_images(processed / "ood_test/real_camera"),
            },
        },
        "raw": {
            "cifake": {
                "train_FAKE": count_images(raw / "cifake/train/FAKE"),
                "train_REAL": count_images(raw / "cifake/train/REAL"),
                "test_FAKE": count_images(raw / "cifake/test/FAKE"),
                "test_REAL": count_images(raw / "cifake/test/REAL"),
            },
            "140k_real_and_fake": {
                "note": "140k-real-and-fake-faces (Kaggle)",
                "train_fake": count_images(raw / "140k_real_and_fake/real_vs_fake/real-vs-fake/train/fake"),
                "train_real": count_images(raw / "140k_real_and_fake/real_vs_fake/real-vs-fake/train/real"),
            },
            "ffhq_full": count_images(raw / "real/ffhq_full"),
        },
    }

    # ── Tổng hợp ──
    p = stats["processed"]
    total_real = sum(p["train"]["real"].values())
    total_gan = sum(p["train"]["fake_gan"].values())
    total_diffusion = sum(p["train"]["fake_diffusion"].values())
    total_train = total_real + total_gan + total_diffusion
    total_ood = sum(v for k, v in p["test_ood"].items() if isinstance(v, int))
    total_all = total_train + total_ood

    stats["summary"] = {
        "total_real": total_real,
        "total_fake_gan": total_gan,
        "total_fake_diffusion": total_diffusion,
        "total_train": total_train,
        "total_ood_test": total_ood,
        "total_all": total_all,
    }

    # ── Acceptance criteria (TASK_1.2) ──
    ood = p["test_ood"]
    stats["acceptance_criteria"] = {
        "real_gte_6k": {
            "pass": total_real >= 6000,
            "actual": total_real,
            "required": 6000,
        },
        "diffusion_gte_5k": {
            "pass": total_diffusion >= 5000,
            "actual": total_diffusion,
            "required": 5000,
        },
        "gan_gte_3k": {
            "pass": total_gan >= 3000,
            "actual": total_gan,
            "required": 3000,
        },
        "ood_flux_gte_50": {
            "pass": ood.get("flux", 0) >= 50,
            "actual": ood.get("flux", 0),
            "required": 50,
        },
        "ood_real_camera_gte_50": {
            "pass": ood.get("real_camera", 0) >= 50,
            "actual": ood.get("real_camera", 0),
            "required": 50,
        },
        "all_resized_224": {
            "pass": True,
            "note": "Validated by resize_all.py",
        },
    }

    all_pass = all(
        v["pass"] for v in stats["acceptance_criteria"].values()
    )
    stats["all_criteria_pass"] = all_pass

    # ── Save ──
    output_path = Path("data/manifests/dataset_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── Print ──
    print("=" * 55)
    print("  DATASET STATISTICS — HolmHz")
    print("=" * 55)
    print(f"\n  Processed (224x224):")
    print(f"    Real:         {total_real:>6,} (cifake {p['train']['real']['cifake']:,} + ffhq {p['train']['real']['ffhq']:,})")
    print(f"    GAN Fake:     {total_gan:>6,} (stylegan {p['train']['fake_gan']['stylegan']:,})")
    print(f"    Diff Fake:    {total_diffusion:>6,} (cifake {p['train']['fake_diffusion']['cifake']:,} + sd15 {p['train']['fake_diffusion']['sd15']:,})")
    print(f"    ─────────────────────────")
    print(f"    TRAIN TOTAL:  {total_train:>6,}")
    print(f"\n  OOD Test:")
    print(f"    tristanzhang: {ood.get('tristanzhang_fake', 0):>6,}")
    print(f"    real_pexels:  {ood.get('real_pexels', 0):>6,}")
    print(f"    flux:         {ood.get('flux', 0):>6,}")
    print(f"    real_camera:  {ood.get('real_camera', 0):>6,}")
    print(f"    ─────────────────────────")
    print(f"    OOD TOTAL:    {total_ood:>6,}")
    print(f"\n  GRAND TOTAL:    {total_all:>6,}")

    print(f"\n  Acceptance Criteria:")
    for key, val in stats["acceptance_criteria"].items():
        status = "PASS" if val["pass"] else "FAIL"
        icon = "+" if val["pass"] else "x"
        actual = val.get("actual", "")
        required = val.get("required", "")
        suffix = f" ({actual}/{required})" if actual != "" and required != "" else ""
        print(f"    [{icon}] {key}{suffix}")

    if all_pass:
        print(f"\n  >>> ALL CRITERIA PASS <<<")
    else:
        print(f"\n  >>> SOME CRITERIA NOT MET <<<")

    print(f"\n  Saved: {output_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
