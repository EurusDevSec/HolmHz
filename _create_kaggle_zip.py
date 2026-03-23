"""Create holmhz-data-v3.zip for Kaggle upload."""
import zipfile
import os
from pathlib import Path
import time

ROOT = Path(".")
OUTPUT = ROOT / "holmhz-data-v3.zip"

# Directories to include (recursively)
INCLUDE_DIRS = [
    "data/processed/train",
    "data/processed/ood_test",
    "data/manifests",
    "src",
    "configs",
    "preprocessing",
]

# Individual files to include
INCLUDE_FILES = [
    "pyproject.toml",
    "scripts/train.py",
    "scripts/test.py",
    "scripts/explain.py",
    "scripts/export_onnx.py",
]

# Extensions to skip inside included dirs
SKIP_EXTENSIONS = {".pyc", ".pyo"}
SKIP_DIRS = {"__pycache__", ".git", ".DS_Store"}

start = time.time()
count = 0

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

with zipfile.ZipFile(OUTPUT, "w") as zf:
    # Add directories
    for dir_path in INCLUDE_DIRS:
        p = ROOT / dir_path
        if not p.exists():
            print(f"  SKIP (not found): {dir_path}")
            continue
        for root, dirs, files in os.walk(p):
            # Filter out skip dirs
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in SKIP_EXTENSIONS:
                    continue
                full = Path(root) / f
                arcname = str(full.relative_to(ROOT).as_posix())
                # Images already compressed → ZIP_STORED (fast)
                # Text/code → ZIP_DEFLATED (smaller)
                compress = zipfile.ZIP_STORED if ext in IMAGE_EXTS else zipfile.ZIP_DEFLATED
                zf.write(full, arcname, compress_type=compress)
                count += 1
                if count % 5000 == 0:
                    elapsed = time.time() - start
                    print(f"  {count} files added ({elapsed:.0f}s)...")

    # Add individual files
    for fpath in INCLUDE_FILES:
        fp = ROOT / fpath
        if fp.exists():
            zf.write(fp, fpath, compress_type=zipfile.ZIP_DEFLATED)
            count += 1
        else:
            print(f"  SKIP (not found): {fpath}")

elapsed = time.time() - start
size_mb = OUTPUT.stat().st_size / 1e6
print(f"\nDone! {count} files → {OUTPUT} ({size_mb:.1f} MB) in {elapsed:.0f}s")
