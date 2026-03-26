# Kaggle Training Guide — v7 (Data Reset + Diffusion Fakes)

Hướng dẫn train EfficientNet-B0 v7 trên Kaggle T4 GPU.

## Dataset v7

| Source | Real | Fake | Type |
|--------|------|------|------|
| rvf10k | 5,000 | 5,000 | Diverse content |
| ciplab_faces | 1,930 | 1,688 | GAN faces |
| DALL-E | — | 2,000 | 🆕 Diffusion |
| Stable Diffusion | — | 2,098 | 🆕 Diffusion |
| Midjourney | — | 930 | 🆕 Diffusion |
| deepfake_real | 5,890 | — | 🆕 Diverse real |
| camera_vs_ai | 234 | 220 | OOD test only |
| **Total** | **13,286** | **12,168** | **25,454** |

---

## Bước 1: Nén & Upload

```bash
# Đã tạo sẵn:
# holmhz-code-v7.zip  (0.3 MB)  — src, configs, scripts, manifests
# holmhz-images-v7.zip (1.3 GB) — tất cả ảnh

# Upload cả 2 file lên Kaggle → New Dataset
# Có thể upload chung 1 dataset hoặc 2 dataset riêng
```

---

## Bước 2: Tạo Kaggle Notebook

### Cell 1: Install & Check GPU

```python
!pip install -q wandb omegaconf timm albumentations tqdm python-dotenv rich scipy scikit-learn

import os, torch
os.environ['WANDB_API_KEY'] = 'YOUR_KEY'  # ← thay key hoặc bỏ qua

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 2: Copy Data & Code

```python
import shutil, os
from pathlib import Path

# === TÌM DATASET TRONG /kaggle/input ===
# Scan tất cả thư mục input để tìm code (có src/holmhz) và images (có raw_v2 hoặc rvf10k)
CODE_INPUT = None
IMG_INPUT = None

for root_dir in Path("/kaggle/input").iterdir():
    if not root_dir.is_dir():
        continue
    # Tìm code: có thư mục src/holmhz
    for p in root_dir.rglob("holmhz"):
        if p.is_dir() and (p.parent.name == "src"):
            CODE_INPUT = p.parent.parent  # thư mục chứa src/
            break
    # Tìm images: có thư mục rvf10k hoặc raw_v2
    for p in root_dir.rglob("rvf10k"):
        if p.is_dir():
            # IMG_INPUT = parent chứa rvf10k
            # Nếu rvf10k nằm trong data/raw_v2/rvf10k → IMG_INPUT = thư mục chứa data/
            # Nếu rvf10k nằm trực tiếp → IMG_INPUT = parent
            IMG_INPUT = p.parent
            break

if CODE_INPUT is None:
    print("❌ Code dataset not found!")
    print("Available in /kaggle/input:")
    for p in sorted(Path("/kaggle/input").rglob("*"))[:50]:
        print(f"  {p}")
    raise FileNotFoundError("Code dataset not found")

if IMG_INPUT is None:
    print("❌ Image dataset not found!")
    for p in sorted(Path("/kaggle/input").rglob("*"))[:50]:
        print(f"  {p}")
    raise FileNotFoundError("Image dataset not found")

print(f"✅ Code: {CODE_INPUT}")
print(f"✅ Images: {IMG_INPUT}")

# === COPY CODE ===
for folder in ["src", "configs", "scripts"]:
    src_path = CODE_INPUT / folder
    if src_path.exists():
        dst = Path(folder)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_path, dst)
        print(f"  Copied {folder}/")

# === COPY MANIFESTS ===
manifest_src = CODE_INPUT / "data" / "manifests_v2"
manifest_dst = Path("data/manifests_v2")
manifest_dst.parent.mkdir(parents=True, exist_ok=True)
if manifest_dst.exists():
    shutil.rmtree(manifest_dst)
shutil.copytree(manifest_src, manifest_dst)
print(f"  Copied data/manifests_v2/")

# === LINK/COPY IMAGES ===
# Tạo symlinks cho TẤT CẢ thư mục ảnh (bao gồm diffusion_fakes!)
img_dst_root = Path("data/raw_v2")
img_dst_root.mkdir(parents=True, exist_ok=True)

# Danh sách TẤT CẢ thư mục cần link
IMAGE_DIRS = ["rvf10k", "ciplab_faces", "camera_vs_ai", "diffusion_fakes", "deepdetect2025"]

for d in IMAGE_DIRS:
    src_dir = IMG_INPUT / d
    dst_dir = img_dst_root / d
    if src_dir.exists() and not dst_dir.exists():
        try:
            os.symlink(str(src_dir), str(dst_dir))
            print(f"  Linked {d}/")
        except OSError:
            # Kaggle có thể không cho phép symlink → fallback copy
            shutil.copytree(src_dir, dst_dir)
            print(f"  Copied {d}/ (symlink failed)")
    elif not src_dir.exists():
        print(f"  ⚠️ Not found: {src_dir}")

# Verify — đếm ảnh
total_images = 0
for d in IMAGE_DIRS:
    p = img_dst_root / d
    if p.exists():
        count = sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        total_images += count
        print(f"  {d}: {count} images")
print(f"\n✅ Total images: {total_images}")
```

### Cell 3: Fix Manifest Paths (Kaggle)

```python
import json
from pathlib import Path

# Manifests có absolute Windows paths → cần fix thành relative Kaggle paths
fixed_total = 0
errors = []

for manifest_name in ["train.json", "val.json", "test_id.json", "test_ood.json"]:
    path = Path(f"data/manifests_v2/{manifest_name}")
    data = json.load(open(path))
    fixed = 0

    for entry in data:
        old_path = entry["path"]
        if "raw_v2" in old_path:
            # Tìm "raw_v2" và giữ từ đó trở đi
            idx = old_path.find("raw_v2")
            rel = old_path[idx:]
            # Fix separators: Windows \ → /
            rel = rel.replace("\\\\", "/").replace("\\", "/")
            new_path = "data/" + rel
            entry["path"] = new_path
            fixed += 1

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {manifest_name}: fixed {fixed}/{len(data)} paths")
    fixed_total += fixed

# Verify — kiểm tra NHIỀU samples, không chỉ 1 (có thể sample đầu tiên OK nhưng sample khác lỗi)
print(f"\n  Verifying paths...")
train_data = json.load(open("data/manifests_v2/train.json"))

# Kiểm tra 20 samples ngẫu nhiên
import random
random.seed(42)
check_samples = random.sample(train_data, min(20, len(train_data)))

ok = 0
for s in check_samples:
    p = Path(s['path'])
    if p.exists():
        ok += 1
    else:
        errors.append(s['path'])

if errors:
    print(f"  ❌ {len(errors)} paths not found:")
    for e in errors[:5]:
        print(f"    {e}")
    print("  Trying to debug...")
    # Show what actually exists
    for d in ["rvf10k", "ciplab_faces", "camera_vs_ai", "diffusion_fakes"]:
        dp = Path(f"data/raw_v2/{d}")
        if dp.exists():
            subdirs = [x.name for x in dp.iterdir() if x.is_dir()]
            print(f"  data/raw_v2/{d}/ contains: {subdirs[:10]}")
else:
    print(f"  ✅ All {ok}/{ok} checked paths exist!")

print(f"\n✅ Manifests fixed! ({fixed_total} paths total)")
```

### Cell 4: Setup Python Path

```python
import sys, os
sys.path.insert(0, "src")
os.environ["PYTHONPATH"] = "src"

import holmhz
print(f"HolmHz: {holmhz.__version__}")

from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors
print(f"Available detectors: {DETECTOR_REGISTRY.list()}")

# Verify data counts
import json
train_data = json.load(open("data/manifests_v2/train.json"))
print(f"Train: {len(train_data)} samples")
fake = sum(1 for d in train_data if d["label"] == 1)
print(f"  Real: {len(train_data) - fake}, Fake: {fake} ({fake/len(train_data)*100:.1f}%)")

# Show sources
sources = {}
for d in train_data:
    sources[d['source']] = sources.get(d['source'], 0) + 1
for s, c in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c}")
```

### Cell 5: Write v7 Config

```python
config_v7 = """
model:
  name: efficientnet_b0
  pretrained: true
  num_classes: 1
  dropout: 0.3
  freeze_backbone: false

training:
  epochs: 30
  batch_size: 32
  learning_rate: 0.0003
  optimizer: adamw
  weight_decay: 0.01
  scheduler: cosine
  pos_weight: 1.0
  early_stopping:
    patience: 7
    monitor: val_auc

data:
  train_manifest: data/manifests_v2/train.json
  val_manifest: data/manifests_v2/val.json
  image_size: 224
  num_workers: 4
  augmentation: true
  use_weighted_sampler: true

wandb:
  project: holmhz
  entity: null
  log_every_n_steps: 10
"""

import os
os.makedirs("configs", exist_ok=True)
with open("configs/train_v7.yaml", "w") as f:
    f.write(config_v7)
print("✅ configs/train_v7.yaml written")
```

### Cell 6: Train!

```python
import os
# Clean old checkpoints
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)

os.makedirs("outputs/checkpoints", exist_ok=True)
print("Training EfficientNet-B0 v7 (with diffusion fakes)...")
!PYTHONPATH=src python scripts/train.py configs/train_v7.yaml data.num_workers=4
```

### Cell 7: Save + Evaluate

```python
import shutil, json, os
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_v7.pt")
    print(f"✅ Saved /kaggle/working/best_v7.pt ({best.stat().st_size/1e6:.1f}MB)")
else:
    print("❌ best.pt not found!")

# Write test config
test_config = """
model:
  name: efficientnet_b0
  checkpoint: outputs/checkpoints/best.pt
  dropout: 0.3

data:
  test_manifest: data/manifests_v2/test_id.json
  ood_manifest: data/manifests_v2/test_ood.json
  image_size: 224
  batch_size: 32
  num_workers: 4

evaluation:
  metrics: [auc, accuracy, f1, precision, recall]
  threshold: 0.5
  save_predictions: true
  output_dir: outputs/evaluation_v7

wandb:
  project: holmhz
  log_results: false
"""
with open("configs/test_v7.yaml", "w") as f:
    f.write(test_config)
print("✅ configs/test_v7.yaml written")

# Full evaluation
print("\n=== Full Evaluation (ID + OOD) ===")
!PYTHONPATH=src python scripts/test.py configs/test_v7.yaml
```

---

## Estimates

| Metric | Estimate |
|--------|----------|
| Training time | ~45-75 phút (30 epochs, 20K images) |
| VRAM | ~4GB (EfficientNet-B0, batch=32) |
| Upload size | code 0.3MB + images 1.3GB |
| Target Val AUC | >0.99 |
| Target OOD camera_ai | >70% (was 2.7% in v6) |
