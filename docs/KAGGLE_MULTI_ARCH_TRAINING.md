# Kaggle Training Guide — Multi-Architecture Benchmark

Hướng dẫn train 3 backbone mới (ResNet-18, ViT-Small/16, Swin-T) trên Kaggle T4 GPU.

## Prerequisites

- Dataset đã upload lên Kaggle: `holmhz-data-v3`
- Code mới nhất đã push lên dataset (bao gồm `timm_backbone.py`, `timm_detector.py`)

---

## Notebook Setup (mỗi model 1 notebook)

### Cell 1: Install & Check GPU

```python
!pip install -q wandb omegaconf timm albumentations tqdm python-dotenv rich scipy scikit-learn

import os
os.environ['WANDB_API_KEY'] = 'YOUR_KEY'  # ← thay key của bạn, hoặc bỏ qua

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 2: Copy Data

```python
import shutil, os
from pathlib import Path

# Auto-detect dataset path
DATA_INPUT = None
for p in Path("/kaggle/input").rglob("src"):
    if p.is_dir() and (p.parent / "data").exists():
        DATA_INPUT = p.parent
        break

if DATA_INPUT is None:
    print("❌ Không tìm được dataset!")
    for p in sorted(Path("/kaggle/input").rglob("*"))[:50]:
        print(p)
    raise FileNotFoundError("Dataset not found")

print(f"✅ Found dataset: {DATA_INPUT}")

# Copy data + src
for folder in ["data", "src", "scripts", "configs", "preprocessing"]:
    src = DATA_INPUT / folder
    if src.exists():
        if Path(folder).exists():
            shutil.rmtree(folder)
        shutil.copytree(src, folder)
        print(f"  Copied {folder}/")
```

### Cell 3: Rebuild Manifests (nếu cần)

```python
import subprocess, json

# Rebuild manifests
result = subprocess.run(
    ["python", "preprocessing/build_splits.py"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("build_splits.py failed")

# Verify
train_data = json.load(open("data/manifests/train.json"))
print(f"Train: {len(train_data)} samples")
assert len(train_data) >= 20000
print("✅ Manifests OK")
```

### Cell 4: Python Path

```python
import sys, os
sys.path.insert(0, "src")
os.environ["PYTHONPATH"] = "src"

import holmhz
print(f"HolmHz: {holmhz.__version__}")

# Verify registry
from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors
print(f"Available detectors: {DETECTOR_REGISTRY.list()}")
```

---

## Train ResNet-18

### Cell 5a: Write Config

```python
config_resnet18 = """
model:
  name: resnet18
  pretrained: true
  num_classes: 1
  dropout: 0.3
  freeze_backbone: false

training:
  epochs: 30
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  pos_weight: 1.2
  early_stopping:
    patience: 10
    monitor: val_auc

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  image_size: 224
  num_workers: 4
  augmentation: true
  use_weighted_sampler: true

wandb:
  project: holmhz
  entity: null
  log_every_n_steps: 10
"""

os.makedirs("configs", exist_ok=True)
with open("configs/train_resnet18.yaml", "w") as f:
    f.write(config_resnet18)
print("✅ configs/train_resnet18.yaml written")
```

### Cell 6a: Train

```python
import os, shutil
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)

os.makedirs("outputs/checkpoints", exist_ok=True)
print("Training ResNet-18...")
!PYTHONPATH=src python scripts/train.py configs/train_resnet18.yaml data.num_workers=4
```

### Cell 7a: Save

```python
import shutil
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_resnet18.pt")
    print("✅ Saved /kaggle/working/best_resnet18.pt")

# Quick eval
!PYTHONPATH=src python scripts/test.py model.name=resnet18 model.checkpoint=outputs/checkpoints/best.pt
```

---

## Train ViT-Small/16

### Cell 5b: Write Config

```python
config_vit_small = """
model:
  name: vit_small
  pretrained: true
  num_classes: 1
  dropout: 0.3
  freeze_backbone: false

training:
  epochs: 30
  batch_size: 16
  learning_rate: 0.0001
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  pos_weight: 1.2
  early_stopping:
    patience: 10
    monitor: val_auc

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  image_size: 224
  num_workers: 4
  augmentation: true
  use_weighted_sampler: true

wandb:
  project: holmhz
  entity: null
  log_every_n_steps: 10
"""

with open("configs/train_vit_small.yaml", "w") as f:
    f.write(config_vit_small)
print("✅ configs/train_vit_small.yaml written")
```

### Cell 6b: Train

```python
import os
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)

print("Training ViT-Small/16...")
!PYTHONPATH=src python scripts/train.py configs/train_vit_small.yaml data.num_workers=4
```

### Cell 7b: Save

```python
import shutil
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_vit_small.pt")
    print("✅ Saved /kaggle/working/best_vit_small.pt")

!PYTHONPATH=src python scripts/test.py model.name=vit_small model.checkpoint=outputs/checkpoints/best.pt
```

---

## Train Swin-T

### Cell 5c: Write Config

```python
config_swin_tiny = """
model:
  name: swin_tiny
  pretrained: true
  num_classes: 1
  dropout: 0.3
  freeze_backbone: false

training:
  epochs: 30
  batch_size: 16
  learning_rate: 0.0001
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  pos_weight: 1.2
  early_stopping:
    patience: 10
    monitor: val_auc

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  image_size: 224
  num_workers: 4
  augmentation: true
  use_weighted_sampler: true

wandb:
  project: holmhz
  entity: null
  log_every_n_steps: 10
"""

with open("configs/train_swin_tiny.yaml", "w") as f:
    f.write(config_swin_tiny)
print("✅ configs/train_swin_tiny.yaml written")
```

### Cell 6c: Train

```python
import os
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)

print("Training Swin-T...")
!PYTHONPATH=src python scripts/train.py configs/train_swin_tiny.yaml data.num_workers=4
```

### Cell 7c: Save

```python
import shutil
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_swin_tiny.pt")
    print("✅ Saved /kaggle/working/best_swin_tiny.pt")

!PYTHONPATH=src python scripts/test.py model.name=swin_tiny model.checkpoint=outputs/checkpoints/best.pt
```

---

## Lưu ý

| Model        | batch_size | VRAM est. | Training time est. (30 epochs) |
| ------------ | ---------- | --------- | ------------------------------ |
| ResNet-18    | 32         | ~4 GB     | ~25 phút                       |
| ViT-Small/16 | 16         | ~8 GB     | ~45 phút                       |
| Swin-T       | 16         | ~10 GB    | ~50 phút                       |

- **Nếu OOM**: giảm `batch_size` xuống 8
- **Nếu hết quota Kaggle**: train 1 model/notebook, mỗi notebook 1 session
- Sau khi train xong, download file `.pt` từ `/kaggle/working/` về local → đặt vào `weights/`
- Tất cả train trên cùng dataset (21,210 samples), cùng hyperparams, fair comparison
