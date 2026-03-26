# Kaggle v9 Training Guide — Dual Approach

## Overview

v9 có 2 approach chạy **cùng dataset** (28K images from v8):

| | Hướng A: EfficientNet + JPEG Aug | Hướng B: CLIP Linear Probe |
|---|---|---|
| Config | `configs/train_v9.yaml` | `configs/train_v9_clip.yaml` |
| Time | ~60-90 min | ~15-30 min |
| Trainable params | 4M (full model) | 769 (linear only!) |
| Fix target | JPEG compression bias | Domain generalization |
| Extra deps | None | `open-clip-torch` |

---

## Cell 1: Setup (Giống v8)

```python
import subprocess, shutil, zipfile
from pathlib import Path

# Unzip code
CODE_ZIP = list(Path("/kaggle/input").rglob("holmhz-code*.zip"))
assert CODE_ZIP, "Upload holmhz-code-v8.zip to dataset!"
with zipfile.ZipFile(CODE_ZIP[0], 'r') as z:
    z.extractall(".")
print("✅ Code extracted")

# Install deps
subprocess.run(["pip", "install", "-q", "albumentations", "omegaconf", "timm"], check=True)
print("✅ Dependencies installed")
```

## Cell 2: Data Setup (Giống v8)

```python
from pathlib import Path

IMG_INPUT = Path("/kaggle/input")
img_dst_root = Path("data/raw_v2")
img_dst_root.mkdir(parents=True, exist_ok=True)

IMAGE_DIRS = ["rvf10k", "ciplab_faces", "camera_vs_ai", "diffusion_fakes", "deepdetect2025"]

for d in IMAGE_DIRS:
    src_dir = IMG_INPUT / d
    if not src_dir.exists():
        for ds_folder in IMG_INPUT.iterdir():
            candidate = ds_folder / d
            if candidate.exists():
                src_dir = candidate
                break
    
    dst_dir = img_dst_root / d
    if dst_dir.exists() or dst_dir.is_symlink():
        dst_dir.unlink() if dst_dir.is_symlink() else shutil.rmtree(dst_dir)
    
    if src_dir.exists():
        dst_dir.symlink_to(src_dir)
        count = sum(1 for _ in src_dir.rglob("*") if _.is_file())
        print(f"  ✅ {d}: {count} files → {dst_dir}")
    else:
        print(f"  ⚠️ {d}: NOT FOUND")

print("✅ Image dirs linked")
```

## Cell 3: Verify Manifests

```python
import json
from pathlib import Path

for m in ["train.json", "val.json", "test_id.json", "test_ood.json"]:
    data = json.load(open(f"data/manifests_v2/{m}"))
    
    # Fix paths: absolute → relative
    fixed = 0
    for item in data:
        p = item["path"]
        if "raw_v2" in p:
            idx = p.find("data/raw_v2")
            if idx >= 0:
                item["path"] = p[idx:]
                fixed += 1
    
    json.dump(data, open(f"data/manifests_v2/{m}", "w"))
    print(f"  {m}: fixed {fixed}/{len(data)} paths")

sample = json.load(open("data/manifests_v2/train.json"))[0]
print(f"\n  Sample path: {sample['path']}")
assert Path(sample['path']).exists(), f"Path not found: {sample['path']}"
print("✅ Manifests verified!")
```

---

## Cell 4A: Train EfficientNet v9 (Hướng A — JPEG Aug Fix)

```python
import subprocess, sys
sys.path.insert(0, "src")

print("Training EfficientNet-B0 v9 (JPEG augmentation fix)...")
result = subprocess.run(
    [sys.executable, "scripts/train.py", "configs/train_v9.yaml"],
    capture_output=False,
)
print(f"Exit code: {result.returncode}")
```

## Cell 4B: Train CLIP v9 (Hướng B — Linear Probe)

```python
import subprocess, sys

# Install open-clip
subprocess.run(["pip", "install", "-q", "open-clip-torch"], check=True)
print("✅ open-clip installed")

sys.path.insert(0, "src")

print("Training CLIP ViT-L/14 Linear Probe...")
result = subprocess.run(
    [sys.executable, "scripts/train.py", "configs/train_v9_clip.yaml"],
    capture_output=False,
)
print(f"Exit code: {result.returncode}")
```

## Cell 5: Save & Download

```python
import shutil, torch
from pathlib import Path

# Save best checkpoint
best = Path("outputs/checkpoints/best.pt")
if best.exists():
    # For EfficientNet v9:
    shutil.copy(best, "/kaggle/working/best_v9.pt")
    size = best.stat().st_size / 1e6
    print(f"✅ Saved /kaggle/working/best_v9.pt ({size:.1f}MB)")
    
    # Or for CLIP v9:
    # shutil.copy(best, "/kaggle/working/best_v9_clip.pt")
```

## Cell 6: Quick Evaluation

```python
import sys, json, subprocess
sys.path.insert(0, "src")

# Write test config  
import yaml
test_config = {
    "model": {"name": "efficientnet_b0", "checkpoint": "outputs/checkpoints/best.pt"},
    "data": {
        "test_manifest": "data/manifests_v2/test_id.json",
        "ood_manifest": "data/manifests_v2/test_ood.json",
        "image_size": 224,
    },
    "evaluation": {"batch_size": 32, "num_workers": 4},
}
with open("configs/test_v9.yaml", "w") as f:
    yaml.dump(test_config, f)

result = subprocess.run(
    [sys.executable, "scripts/evaluate.py", "configs/test_v9.yaml"],
    capture_output=False,
)
```

---

## Notes

- **Hướng A** (EfficientNet): Dùng cùng dataset v8, chỉ thay đổi augmentation.
  Retrain với JPEG compression p=0.7, quality 50-95. Fix iPhone/Facebook false positive.

- **Hướng B** (CLIP): Cần `pip install open-clip-torch` (thêm ~1GB download).  
  CLIP ViT-L/14 tự dynamic download weights (~1.7GB).
  Chỉ train 769 params → converge rất nhanh (~15 min).

- **Recommend**: Chạy Hướng A trước (không cần thêm deps), download `best_v9.pt` test trước.
  Sau đó chạy Hướng B, download `best_v9_clip.pt` so sánh.
