# Kaggle Training Guide — Retrain 3 Models trên Dataset v2

Hướng dẫn retrain ResNet-18, ViT-Small/16, Swin-T trên Kaggle T4 GPU với dataset v2 (raw_v2).  
Mục tiêu: benchmark công bằng với EfficientNet-B0 v7 (đã train trên cùng dataset này).

## Tại sao cần retrain?

3 models trước đó train trên raw (dataset cũ, chưa chuẩn hóa).  
EfficientNet-B0 train trên raw_v2 (dataset v2, chuẩn). So sánh OOD không công bằng.  
→ Retrain cả 3 trên raw_v2 với **cùng hyperparams**.

## Kaggle Datasets đã có

| Dataset Kaggle       | Nội dung                                     | Size    |
| -------------------- | -------------------------------------------- | ------- |
| **holmhz-images-v8** | Ảnh raw_v2 (rvf10k, ciplab, diffusion, ...)  | ~1.3 GB |
| **holmhz-code-v11**  | src/, configs/, scripts/, data/manifests_v2/ | ~0.3 MB |

> **Lưu ý**: Code v11 có thêm freq_detector, clip_detector, exif_analyzer — KHÔNG ảnh hưởng.  
> Chỉ cần `timm_detector.py` + `timm_backbone.py` là đủ cho 3 models.

## Dataset v2 — raw_v2 (35,272 ảnh tổng)

### Cấu trúc thư mục

```
data/raw_v2/
├── rvf10k/          # Real vs Fake 10K (train + valid)
├── ciplab_faces/    # CIPLab real & fake faces
├── diffusion_fakes/ # DALL-E, SD, Midjourney, StyleGAN, Real, ...
├── deepdetect2025/  # DeepDetect 2025 (real + fake)
└── camera_vs_ai/    # Camera real vs AI photos (OOD test)
```

### Phân bổ theo source (manifests_v2)

| Source                   | Count | Label | Type             |
| ------------------------ | ----- | ----- | ---------------- |
| deepfake_collection_real | 4,712 | Real  | Diverse real     |
| dd2025_real              | 4,000 | Real  | DeepDetect real  |
| dd2025_fake              | 4,000 | Fake  | DeepDetect fake  |
| rvf10k_train_real        | 2,800 | Real  | RvF10K           |
| rvf10k_train_fake        | 2,800 | Fake  | RvF10K           |
| ciplab_training_real     | 1,730 | Real  | CIPLab faces     |
| sd_fake                  | 1,680 | Fake  | Stable Diffusion |
| dalle_fake               | 1,600 | Fake  | DALL-E           |
| ciplab_training_fake     | 1,536 | Fake  | CIPLab GAN faces |
| rvf10k_valid_real        | 1,200 | Real  | RvF10K valid     |
| rvf10k_valid_fake        | 1,200 | Fake  | RvF10K valid     |
| midjourney_fake          | 744   | Fake  | Midjourney       |
| camera_train_real        | 112   | Real  | Camera photos    |
| camera_train_ai          | 106   | Fake  | AI photos        |

### Splits (manifests_v2)

| Split        | Total  | Real   | Fake   | File                              |
| ------------ | ------ | ------ | ------ | --------------------------------- |
| **Train**    | 28,220 | 14,554 | 13,666 | `data/manifests_v2/train.json`    |
| **Val**      | 3,526  | 1,819  | 1,707  | `data/manifests_v2/val.json`      |
| **Test ID**  | 3,526  | 1,819  | 1,707  | `data/manifests_v2/test_id.json`  |
| **Test OOD** | 182    | 94     | 88     | `data/manifests_v2/test_ood.json` |

> OOD test: 182 ảnh từ camera_vs_ai (camera_real 94 + camera_ai 88)

---

## Tạo Kaggle Notebook

### Cell 1: Install & Check GPU

```python
!pip install -q wandb omegaconf timm albumentations tqdm python-dotenv rich scipy scikit-learn

import os, torch
os.environ['WANDB_API_KEY'] = 'YOUR_KEY'  # ← thay key hoặc bỏ qua wandb

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 2: Copy Data & Code

```python
import shutil, os, json
from pathlib import Path

# === TÌM DATASET TRONG /kaggle/input ===
CODE_INPUT = None
IMG_INPUT = None

for root_dir in Path("/kaggle/input").iterdir():
    if not root_dir.is_dir():
        continue

    # Tìm code: có thư mục src/holmhz
    for p in root_dir.rglob("holmhz"):
        if p.is_dir() and p.parent.name == "src":
            CODE_INPUT = p.parent.parent
            break

    # Tìm images: có thư mục rvf10k (đặc trưng raw_v2)
    for p in root_dir.rglob("rvf10k"):
        if p.is_dir():
            IMG_INPUT = p.parent  # parent chứa rvf10k, ciplab_faces, ...
            break

if CODE_INPUT is None:
    print("Code dataset not found!")
    for p in sorted(Path("/kaggle/input").rglob("*"))[:50]:
        print(f"  {p}")
    raise FileNotFoundError("Code dataset not found")

if IMG_INPUT is None:
    print("Image dataset not found!")
    for p in sorted(Path("/kaggle/input").rglob("*"))[:50]:
        print(f"  {p}")
    raise FileNotFoundError("Image dataset not found")

print(f"Code:   {CODE_INPUT}")
print(f"Images: {IMG_INPUT}")

# === COPY CODE ===
for folder in ["src", "configs", "scripts"]:
    src_path = CODE_INPUT / folder
    if src_path.exists():
        dst = Path(folder)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_path, dst)
        print(f"  Copied {folder}/")

# === COPY MANIFESTS_V2 ===
manifest_src = CODE_INPUT / "data" / "manifests_v2"
manifest_dst = Path("data/manifests_v2")
manifest_dst.parent.mkdir(parents=True, exist_ok=True)
if manifest_dst.exists():
    shutil.rmtree(manifest_dst)
shutil.copytree(manifest_src, manifest_dst)
print(f"  Copied data/manifests_v2/")

# === LINK/COPY IMAGES → data/raw_v2/ ===
img_dst_root = Path("data/raw_v2")
img_dst_root.mkdir(parents=True, exist_ok=True)

IMAGE_DIRS = ["rvf10k", "ciplab_faces", "camera_vs_ai", "diffusion_fakes", "deepdetect2025"]

for d in IMAGE_DIRS:
    src_dir = IMG_INPUT / d
    dst_dir = img_dst_root / d
    if src_dir.exists() and not dst_dir.exists():
        try:
            os.symlink(str(src_dir), str(dst_dir))
            print(f"  Linked {d}/")
        except OSError:
            shutil.copytree(src_dir, dst_dir)
            print(f"  Copied {d}/ (symlink failed)")
    elif not src_dir.exists():
        print(f"  Not found: {src_dir}")

# === VERIFY ===
total_images = 0
for d in IMAGE_DIRS:
    p = img_dst_root / d
    if p.exists():
        count = sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        total_images += count
        print(f"  {d}: {count} images")
    else:
        print(f"  {d}: NOT FOUND")
print(f"\nTotal images: {total_images}")
```

### Cell 3: Fix Manifest Paths (Windows → Kaggle)

```python
import json, random
from pathlib import Path

# Manifests chứa absolute Windows paths (R:\_Projects\...\data\raw_v2\...)
# → fix thành relative Kaggle paths (data/raw_v2/...)
fixed_total = 0
errors = []

for manifest_name in ["train.json", "val.json", "test_id.json", "test_ood.json"]:
    path = Path(f"data/manifests_v2/{manifest_name}")
    if not path.exists():
        print(f"  {manifest_name} not found, skipping")
        continue

    data = json.load(open(path))
    fixed = 0

    for entry in data:
        old_path = entry["path"]
        # Nếu đã là relative path hợp lệ → skip
        if old_path.startswith("data/raw_v2/") and Path(old_path).exists():
            continue

        # Fix: tìm "raw_v2" trong path, giữ từ đó
        normalized = old_path.replace("\\\\", "/").replace("\\", "/")
        if "raw_v2" in normalized:
            idx = normalized.find("raw_v2")
            new_path = "data/" + normalized[idx:]
            entry["path"] = new_path
            fixed += 1

    if fixed > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"  {manifest_name}: {len(data)} samples, fixed {fixed} paths")
    fixed_total += fixed

# Verify 20 random samples
random.seed(42)
train_data = json.load(open("data/manifests_v2/train.json"))
check = random.sample(train_data, min(20, len(train_data)))

ok = 0
for s in check:
    if Path(s['path']).exists():
        ok += 1
    else:
        errors.append(s['path'])

if errors:
    print(f"\n  {len(errors)} paths not found:")
    for e in errors[:5]:
        print(f"    {e}")
    # Debug: show actual structure
    for d in Path("data/raw_v2").iterdir():
        if d.is_dir():
            subdirs = [x.name for x in d.iterdir() if x.is_dir()][:5]
            print(f"  data/raw_v2/{d.name}/ -> {subdirs}")
else:
    print(f"\n  All {ok}/{len(check)} checked paths exist!")

print(f"\nManifests fixed! ({fixed_total} paths total)")
```

### Cell 4: Setup Python Path & Verify

```python
import sys, os, json
sys.path.insert(0, "src")
os.environ["PYTHONPATH"] = "src"

import holmhz
print(f"HolmHz: {holmhz.__version__}")

from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors
print(f"Available detectors: {DETECTOR_REGISTRY.list()}")

# Verify data counts
train_data = json.load(open("data/manifests_v2/train.json"))
val_data = json.load(open("data/manifests_v2/val.json"))
ood_data = json.load(open("data/manifests_v2/test_ood.json"))
print(f"\nTrain: {len(train_data)} samples")
print(f"Val:   {len(val_data)} samples")
print(f"OOD:   {len(ood_data)} samples")

# Show source distribution
sources = {}
for d in train_data:
    sources[d['source']] = sources.get(d['source'], 0) + 1
print("\nTrain sources:")
for s, c in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c}")
```

---

## Train ResNet-18

### Cell 5a: Config + Train

```python
import os

# Write config (same hyperparams as EfficientNet-B0)
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

os.makedirs("configs", exist_ok=True)
with open("configs/train_resnet18.yaml", "w") as f:
    f.write(config_resnet18)

# Clean old checkpoints
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)
os.makedirs("outputs/checkpoints", exist_ok=True)

print("=" * 60)
print("Training ResNet-18 (11M params, batch=32)")
print("=" * 60)
!PYTHONPATH=src python scripts/train.py configs/train_resnet18.yaml data.num_workers=4
```

### Cell 5.5: Patch test.py (fix DataParallel key mismatch)

> **Kaggle dùng 2 GPU → DataParallel → checkpoint có prefix `module.`**
> Code `test.py` trên Kaggle (v11) chưa xử lý prefix này. Cell này patch fix.

```python
import re

test_py = "scripts/test.py"
with open(test_py, "r") as f:
    content = f.read()

# Check if already patched
if "module." not in content:
    # Find the line: model.load_state_dict(checkpoint["model_state_dict"])
    # or: model.load_state_dict(state_dict)
    old_pattern = r'(\s+)(state_dict = checkpoint\["model_state_dict"\])'
    patch = r'''\1state_dict = checkpoint["model_state_dict"]
\1# Strip DataParallel 'module.' prefix if present
\1if any(k.startswith("module.") for k in state_dict):
\1    print("  ↳ Stripping DataParallel 'module.' prefix from checkpoint keys")
\1    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}'''

    if re.search(old_pattern, content):
        content = re.sub(old_pattern, patch, content)
        with open(test_py, "w") as f:
            f.write(content)
        print("✅ Patched test.py — DataParallel module. prefix will be stripped")
    else:
        # Fallback: patch the load_state_dict line directly
        old_load = 'model.load_state_dict(checkpoint["model_state_dict"])'
        new_load = '''state_dict = checkpoint["model_state_dict"]
    # Strip DataParallel 'module.' prefix if present
    if any(k.startswith("module.") for k in state_dict):
        print("  ↳ Stripping DataParallel 'module.' prefix from checkpoint keys")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)'''
        if old_load in content:
            content = content.replace(old_load, new_load)
            with open(test_py, "w") as f:
                f.write(content)
            print("✅ Patched test.py (fallback) — DataParallel module. prefix will be stripped")
        else:
            print("⚠️ Could not find load pattern — manual patch needed")
else:
    print("✅ test.py already handles module. prefix — no patch needed")
```

### Cell 6a: Save + Evaluate ResNet-18

```python
import shutil, os, json
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_resnet18_v2.pt")
    print(f"✅ Saved best_resnet18_v2.pt ({best.stat().st_size/1e6:.1f} MB)")
else:
    print("❌ best.pt not found!")
    raise FileNotFoundError("Training failed — no checkpoint")

# Write test config
test_config = """
model:
  name: resnet18
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
  output_dir: outputs/evaluation_resnet18

wandb:
  project: holmhz
  log_results: false
"""
with open("configs/test_resnet18.yaml", "w") as f:
    f.write(test_config)

print("\n=== ResNet-18 Evaluation (ID + OOD) ===")
!PYTHONPATH=src python scripts/test.py configs/test_resnet18.yaml
```

---

## Train ViT-Small/16

### Cell 5b: Config + Train

```python
import os

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

with open("configs/train_vit_small.yaml", "w") as f:
    f.write(config_vit_small)

# Clean old checkpoints
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)

print("=" * 60)
print("Training ViT-Small/16 (22M params, batch=16)")
print("=" * 60)
!PYTHONPATH=src python scripts/train.py configs/train_vit_small.yaml data.num_workers=4
```

### Cell 6b: Save + Evaluate ViT-Small

```python
import shutil
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_vit_small_v2.pt")
    print(f"✅ Saved best_vit_small_v2.pt ({best.stat().st_size/1e6:.1f} MB)")
else:
    print("❌ best.pt not found!")
    raise FileNotFoundError("Training failed — no checkpoint")

test_config = """
model:
  name: vit_small
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
  output_dir: outputs/evaluation_vit_small

wandb:
  project: holmhz
  log_results: false
"""
with open("configs/test_vit_small.yaml", "w") as f:
    f.write(test_config)

print("\n=== ViT-Small/16 Evaluation (ID + OOD) ===")
!PYTHONPATH=src python scripts/test.py configs/test_vit_small.yaml
```

---

## Train Swin-T

### Cell 5c: Config + Train

```python
import os

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

with open("configs/train_swin_tiny.yaml", "w") as f:
    f.write(config_swin_tiny)

# Clean old checkpoints
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)

print("=" * 60)
print("Training Swin-T (28M params, batch=16)")
print("=" * 60)
!PYTHONPATH=src python scripts/train.py configs/train_swin_tiny.yaml data.num_workers=4
```

### Cell 6c: Save + Evaluate Swin-T

```python
import shutil
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_swin_tiny_v2.pt")
    print(f"✅ Saved best_swin_tiny_v2.pt ({best.stat().st_size/1e6:.1f} MB)")
else:
    print("❌ best.pt not found!")
    raise FileNotFoundError("Training failed — no checkpoint")

test_config = """
model:
  name: swin_tiny
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
  output_dir: outputs/evaluation_swin_tiny

wandb:
  project: holmhz
  log_results: false
"""
with open("configs/test_swin_tiny.yaml", "w") as f:
    f.write(test_config)

print("\n=== Swin-T Evaluation (ID + OOD) ===")
!PYTHONPATH=src python scripts/test.py configs/test_swin_tiny.yaml
```

---

## Cell 7: So sánh kết quả (chạy sau khi train xong cả 3)

```python
import json
from pathlib import Path

print("=" * 70)
print("BENCHMARK COMPARISON — All models trained on Dataset v2")
print("=" * 70)

# Collect results
results = {}
for name, eval_dir in [
    ("ResNet-18", "outputs/evaluation_resnet18"),
    ("ViT-Small/16", "outputs/evaluation_vit_small"),
    ("Swin-T", "outputs/evaluation_swin_tiny"),
]:
    report_path = Path(eval_dir) / "eval_report.json"
    if report_path.exists():
        data = json.load(open(report_path))
        results[name] = data
        print(f"\n--- {name} ---")
        if "id" in data:
            print(f"  ID:  AUC={data['id'].get('auc', 'N/A'):.4f}  Acc={data['id'].get('accuracy', 'N/A'):.4f}")
        if "ood" in data:
            print(f"  OOD: AUC={data['ood'].get('auc', 'N/A'):.4f}  Acc={data['ood'].get('accuracy', 'N/A'):.4f}")
    else:
        print(f"\n--- {name} --- ❌ eval_report.json not found")

# Reference: EfficientNet-B0 v7 (trained on same dataset v2)
print(f"\n--- EfficientNet-B0 v7 (reference, same dataset) ---")
print(f"  ID:  AUC=0.9984  Acc=0.9870")
print(f"  OOD: AUC=~0.44   Acc=0.5385")

print("\n" + "=" * 70)
print("Download checkpoints from /kaggle/working/:")
print("  best_resnet18_v2.pt")
print("  best_vit_small_v2.pt")
print("  best_swin_tiny_v2.pt")
print("=" * 70)
```

---

## Estimates

| Model        | batch_size | VRAM est. | Training time (30 epochs, ~28K images) |
| ------------ | ---------- | --------- | -------------------------------------- |
| ResNet-18    | 32         | ~4 GB     | ~25 phút                               |
| ViT-Small/16 | 16         | ~8 GB     | ~45 phút                               |
| Swin-T       | 16         | ~10 GB    | ~55 phút                               |
| **Tổng**     |            |           | **~2 giờ**                             |

## Troubleshooting

| Vấn đề                          | Giải pháp                                           |
| ------------------------------- | --------------------------------------------------- |
| OOM (Out of Memory)             | Giảm `batch_size` xuống 8                           |
| Path not found                  | Chạy lại Cell 3 để fix manifest paths               |
| Hết quota Kaggle                | Train 1-2 models/session, phiên sau train tiếp      |
| `freq_fft` import error         | Bỏ qua — không dùng frequency detector              |
| `clip_detector` import error    | Bỏ qua — không dùng CLIP                            |
| test.py fail `model.checkpoint` | Kiểm tra best.pt tồn tại trong outputs/checkpoints/ |

## Sau khi train xong

1. Download 3 files `.pt` từ `/kaggle/working/` về local
2. Đặt vào `outputs/checkpoints/`:
   - `best_resnet18_v2.pt`
   - `best_vit_small_v2.pt`
   - `best_swin_tiny_v2.pt`
3. Ghi nhận kết quả ID AUC, OOD AUC, per-source accuracy từ Cell 7
4. So sánh với EfficientNet-B0 v7 (cùng dataset v2)
