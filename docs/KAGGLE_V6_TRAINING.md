# Kaggle Training Guide — v6 Data Reset

Hướng dẫn train EfficientNet-B0 v6 trên Kaggle T4 GPU với dữ liệu sạch (không CIFAKE).

## Thay đổi so với v4/v5

| | v4 (cũ) | v6 (mới) |
|---|---|---|
| Dataset | 21K (46% CIFAKE 32×32) | 14.5K (rvf10k + ciplab + camera_vs_ai) |
| Real diversity | 2 domain (FFHQ + CIFAR) | Multi-domain (objects, scenes, faces, camera) |
| Fake generators | CIFAKE SD1.4 + StyleGAN | Diverse (GAN + Diffusion families) |
| Manifests | `data/manifests/` | `data/manifests_v2/` |
| Config | `train_v5.yaml` | `train_v6.yaml` |
| Backbone | Frozen → Unfroz | Unfrozen (LR=0.0003) |

---

## Bước 1: Nén dữ liệu để upload Kaggle

```bash
# Chạy tại thư mục dự án
cd r:/_Projects/Eurus_Workspace/HolmHz

# Nén CODE (nhỏ, ~5MB)
zip -r holmhz-code-v6.zip \
    src/ configs/ scripts/ data/manifests_v2/ \
    -x "*.pyc" "__pycache__/*" ".git/*" "*.egg-info/*"

# Nén DATA riêng (lớn, ~800MB)
zip -r holmhz-images-v6.zip \
    data/raw_v2/rvf10k/ \
    data/raw_v2/ciplab_faces/ \
    data/raw_v2/camera_vs_ai/
```

> **Lưu ý**: Upload 2 file zip lên Kaggle Datasets:
> 1. `holmhz-code-v6` (code + configs + manifests)
> 2. `holmhz-images-v6` (ảnh training/test)

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
import shutil
from pathlib import Path

# Auto-detect code dataset
CODE_INPUT = None
for p in Path("/kaggle/input").rglob("src"):
    if p.is_dir() and (p / "holmhz").exists():
        CODE_INPUT = p.parent
        break

# Auto-detect image dataset
IMG_INPUT = None
for p in Path("/kaggle/input").rglob("rvf10k"):
    if p.is_dir():
        IMG_INPUT = p.parent
        break

if CODE_INPUT is None:
    raise FileNotFoundError("Code dataset not found! Check Kaggle dataset name.")
if IMG_INPUT is None:
    raise FileNotFoundError("Image dataset not found! Check Kaggle dataset name.")

print(f"✅ Code: {CODE_INPUT}")
print(f"✅ Images: {IMG_INPUT}")

# Copy code
for folder in ["src", "configs", "scripts"]:
    src = CODE_INPUT / folder
    if src.exists():
        if Path(folder).exists():
            shutil.rmtree(folder)
        shutil.copytree(src, folder)
        print(f"  Copied {folder}/")

# Copy manifests
manifest_src = CODE_INPUT / "data" / "manifests_v2"
manifest_dst = Path("data/manifests_v2")
manifest_dst.parent.mkdir(parents=True, exist_ok=True)
if manifest_dst.exists():
    shutil.rmtree(manifest_dst)
shutil.copytree(manifest_src, manifest_dst)
print(f"  Copied data/manifests_v2/")

# Symlink images (save disk space on Kaggle)
img_dst = Path("data/raw_v2")
img_dst.mkdir(parents=True, exist_ok=True)
for d in ["rvf10k", "ciplab_faces", "camera_vs_ai"]:
    src_dir = IMG_INPUT / d
    dst_dir = img_dst / d
    if src_dir.exists() and not dst_dir.exists():
        os.symlink(str(src_dir), str(dst_dir))
        print(f"  Linked {d}/")
```

### Cell 3: Fix Manifest Paths (Kaggle)

```python
import json
from pathlib import Path

# Manifests có absolute paths local → cần fix cho Kaggle
for manifest_name in ["train.json", "val.json", "test_id.json", "test_ood.json"]:
    path = Path(f"data/manifests_v2/{manifest_name}")
    data = json.load(open(path))

    fixed = 0
    for entry in data:
        old_path = entry["path"]
        # Thay absolute Windows path → relative Kaggle path
        # Tìm "data/raw_v2/" trong path và giữ phần sau
        if "raw_v2" in old_path:
            idx = old_path.find("raw_v2")
            new_path = "data/" + old_path[idx:].replace("\\\\", "/").replace("\\", "/")
            entry["path"] = new_path
            fixed += 1

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {manifest_name}: fixed {fixed}/{len(data)} paths")

# Verify
sample = json.load(open("data/manifests_v2/train.json"))[0]
print(f"\n  Sample path: {sample['path']}")
assert Path(sample['path']).exists(), f"Path not found: {sample['path']}"
print("✅ Manifests fixed!")
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

# Verify data
import json
train_data = json.load(open("data/manifests_v2/train.json"))
print(f"Train: {len(train_data)} samples")
fake = sum(1 for d in train_data if d["label"] == 1)
print(f"  Real: {len(train_data) - fake}, Fake: {fake} ({fake/len(train_data)*100:.1f}%)")
```

### Cell 5: Write v6 Config

```python
config_v6 = """
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
with open("configs/train_v6.yaml", "w") as f:
    f.write(config_v6)
print("✅ configs/train_v6.yaml written")
```

### Cell 6: Train!

```python
import os
# Clean old checkpoints
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)

os.makedirs("outputs/checkpoints", exist_ok=True)
print("Training EfficientNet-B0 v6 (data reset)...")
!PYTHONPATH=src python scripts/train.py configs/train_v6.yaml data.num_workers=4
```

### Cell 7: Save + Evaluate

```python
import shutil, json
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_v6.pt")
    print(f"✅ Saved /kaggle/working/best_v6.pt ({best.stat().st_size/1e6:.1f}MB)")

# Quick evaluate on test_id
print("\n=== Test ID ===")
!PYTHONPATH=src python scripts/test.py model.name=efficientnet_b0 \
    model.checkpoint=outputs/checkpoints/best.pt \
    data.test_manifest=data/manifests_v2/test_id.json

# OOD evaluate (camera_vs_ai!)
print("\n=== Test OOD (Camera vs AI) ===")
!PYTHONPATH=src python scripts/test.py model.name=efficientnet_b0 \
    model.checkpoint=outputs/checkpoints/best.pt \
    data.test_manifest=data/manifests_v2/test_ood.json
```

---

## Sau khi train xong

1. Download `best_v6.pt` từ `/kaggle/working/`
2. Copy về dự án: `outputs/checkpoints/best_v6.pt`
3. Update web demo config: thay checkpoint path
4. Re-test trên web demo

---

## Estimates

| Metric | Estimate |
|--------|----------|
| Training time | ~40-60 phút (30 epochs, 11K images) |
| VRAM | ~4GB (EfficientNet-B0, batch=32) |
| Dataset size upload | ~800MB (images) + ~5MB (code) |
| Target Val AUC | >0.90 (vs v4 OOD 0.78) |
