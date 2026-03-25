"""
HolmHz Data v2 — Prepare unified manifests from downloaded datasets.

Sources:
1. rvf10k: 10K images (real diverse + AI-generated), balanced
2. ciplab_faces: ~4K face images (real + StyleGAN/ProGAN)
3. camera_vs_ai: 454 images (camera real + AI-generated diverse)

Creates: data/manifests_v2/{train,val,test_id,test_ood}.json

Each entry: {"path": "abs/path.jpg", "label": 0|1, "source": "name"}
label=0 → Real, label=1 → Fake

Usage:
    source .venv/Scripts/activate
    python scripts/prepare_data_v2.py
"""

import json
import random
import sys
from pathlib import Path

SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_V2 = PROJECT_ROOT / "data" / "raw_v2"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests_v2"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan(folder: Path):
    """Recursively find images."""
    return sorted(f for f in folder.rglob("*") if f.suffix.lower() in EXTENSIONS and f.is_file())


def collect_datasets():
    """Collect all samples from downloaded datasets."""
    samples = []

    # ──────────────────────────────────────
    # 1. rvf10k (10K images, diverse content)
    # ──────────────────────────────────────
    rvf_base = RAW_V2 / "rvf10k" / "rvf10k"
    for split in ["train", "valid"]:
        for label_name, label in [("real", 0), ("fake", 1)]:
            d = rvf_base / split / label_name
            if d.exists():
                imgs = scan(d)
                for img in imgs:
                    samples.append({
                        "path": str(img.resolve()),
                        "label": label,
                        "source": f"rvf10k_{split}_{label_name}",
                        "dataset": "rvf10k",
                    })

    # ──────────────────────────────────────
    # 2. ciplab_faces (~4K face images)
    # ──────────────────────────────────────
    for base in [
        RAW_V2 / "ciplab_faces" / "real_and_fake_face",
        RAW_V2 / "ciplab_faces" / "real_and_fake_face_detection" / "real_and_fake_face",
    ]:
        for label_name, label in [("training_real", 0), ("training_fake", 1)]:
            d = base / label_name
            if d.exists():
                imgs = scan(d)
                for img in imgs:
                    samples.append({
                        "path": str(img.resolve()),
                        "label": label,
                        "source": f"ciplab_{label_name}",
                        "dataset": "ciplab",
                    })

    # ──────────────────────────────────────
    # 3. camera_vs_ai (454 images, camera photos)
    # ──────────────────────────────────────
    cam_base = RAW_V2 / "camera_vs_ai" / "ai vs real photos"
    for label_name, label, dirname in [
        ("camera_real", 0, "Camera_images"),
        ("camera_ai", 1, "Ai_Images"),
    ]:
        d = cam_base / dirname
        if d.exists():
            imgs = scan(d)
            for img in imgs:
                samples.append({
                    "path": str(img.resolve()),
                    "label": label,
                    "source": label_name,
                    "dataset": "camera_vs_ai",
                })

    # ──────────────────────────────────────
    # 4. diffusion_fakes — DALL-E, SD, Midjourney + verified Real
    #    Source: jayanthbottu/labeled-deepfake-image-collection (Kaggle)
    #    Categorized by AI model, peer-reviewed labels
    # ──────────────────────────────────────
    diff_base = RAW_V2 / "diffusion_fakes"
    if diff_base.exists():
        # Diffusion-model fakes (the key addition for OOD fix!)
        for dirname, source_name in [
            ("DALL-E", "dalle_fake"),
            ("Stable Diffusion", "sd_fake"),
            ("Midjourney", "midjourney_fake"),
        ]:
            d = diff_base / dirname
            if d.exists():
                imgs = scan(d)
                for img in imgs:
                    samples.append({
                        "path": str(img.resolve()),
                        "label": 1,
                        "source": source_name,
                        "dataset": "diffusion_fakes",
                    })

        # Extra verified Real images (diverse, non-face)
        d = diff_base / "Real"
        if d.exists():
            imgs = scan(d)
            for img in imgs:
                samples.append({
                    "path": str(img.resolve()),
                    "label": 0,
                    "source": "deepfake_collection_real",
                    "dataset": "diffusion_fakes",
                })

    return samples


def deduplicate(samples):
    """Remove duplicate paths."""
    seen = set()
    unique = []
    for s in samples:
        p = s["path"]
        if p not in seen:
            seen.add(p)
            unique.append(s)
    return unique


def split_data(samples, val_ratio=0.10, test_ratio=0.10, ood_sources=None):
    """Split into train/val/test_id/test_ood.

    OOD test: camera_vs_ai images (never seen during training).
    ID splits: stratified by source.
    """
    if ood_sources is None:
        ood_sources = {"camera_real", "camera_ai"}

    ood = [s for s in samples if s["source"] in ood_sources]
    id_samples = [s for s in samples if s["source"] not in ood_sources]

    random.shuffle(id_samples)

    # Stratify by source
    by_source = {}
    for s in id_samples:
        by_source.setdefault(s["source"], []).append(s)

    train, val, test_id = [], [], []
    for src, items in by_source.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test_id.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test_id)
    random.shuffle(ood)

    return train, val, test_id, ood


def print_stats(name, data):
    """Print split statistics."""
    total = len(data)
    fake = sum(1 for d in data if d["label"] == 1)
    real = total - fake
    sources = {}
    for d in data:
        sources[d["source"]] = sources.get(d["source"], 0) + 1

    print(f"\n  {name}: {total} samples (real={real}, fake={fake}, ratio={fake/total*100:.1f}%)")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        label = "FAKE" if any(d["label"] == 1 and d["source"] == src for d in data) else "REAL"
        print(f"    [{label}] {src}: {count}")


def main():
    print("=" * 60)
    print("HolmHz Data v2 — Build Manifests")
    print("=" * 60)

    # Collect
    samples = collect_datasets()
    samples = deduplicate(samples)
    print(f"\nTotal unique images: {len(samples)}")

    fake = sum(1 for s in samples if s["label"] == 1)
    real = len(samples) - fake
    print(f"  Real: {real}, Fake: {fake}")

    # Check for image readability (sample check)
    from PIL import Image
    bad = 0
    for s in random.sample(samples, min(100, len(samples))):
        try:
            Image.open(s["path"]).convert("RGB")
        except Exception:
            bad += 1
    print(f"  Readability check (100 samples): {100 - bad}/100 OK")

    # Split
    train, val, test_id, test_ood = split_data(samples)

    print_stats("train", train)
    print_stats("val", val)
    print_stats("test_id", test_id)
    print_stats("test_ood (camera_vs_ai)", test_ood)

    # Save manifests
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [
        ("train.json", train),
        ("val.json", val),
        ("test_id.json", test_id),
        ("test_ood.json", test_ood),
    ]:
        # Remove dataset key (not needed for training)
        clean = [{"path": d["path"], "label": d["label"], "source": d["source"]} for d in data]
        path = MANIFEST_DIR / name
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n  Saved: {path} ({len(clean)} samples)")

    print(f"\n{'=' * 60}")
    print("Done! Manifests saved to data/manifests_v2/")
    print("=" * 60)


if __name__ == "__main__":
    main()
