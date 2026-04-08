# Kaggle Training Guide — v11 (FFT Frequency Detector)

Hướng dẫn train `FrequencyDetector` (FFT CNN ~1.7M params) trên Kaggle T4 x2 GPU.
Dùng **cùng dataset v8** — model tự convert RGB → FFT spectrum trong forward pass.

> ⚠️ Cấu trúc cells giống v9 (đã chạy thành công). DataParallel cho T4x2.

## Dataset v11 (Giống v8/v9)

| Source | Real | Fake | Type |
|--------|------|------|------|
| rvf10k | 5,000 | 5,000 | Diverse content |
| ciplab_faces | 1,930 | 1,688 | GAN faces |
| DALL-E | — | 2,000 | Diffusion |
| Stable Diffusion | — | 2,098 | Diffusion |
| Midjourney | — | 930 | Diffusion |
| deepfake_real | 5,890 | — | Diverse real |
| dd2025_real | 5,000 | — | DeepDetect-2025 |
| dd2025_fake | — | 5,000 | DeepDetect-2025 |
| camera_vs_ai (60%) | 112 | 106 | Camera domain (train) |
| camera_vs_ai (40%) | 94 | 88 | OOD test only |
| **Total** | **19,026** | **16,822** | **~35K** |

---

## Bước 1: Nén & Upload

```bash
# Đã tạo sẵn từ v8/v9:
# holmhz-code-v11.zip  — src (có freq_detector.py), configs, scripts, manifests
# holmhz-images-v8.zip — tất cả ảnh (GIỐNG v8/v9, không cần upload lại nếu đã có)

# Upload lên Kaggle → New Dataset
# Nếu đã có holmhz-images-v8 → chỉ cần upload code mới
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
n_gpus = torch.cuda.device_count()
print(f"GPU count: {n_gpus}")
for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"  GPU {i}: {name} ({mem:.1f}GB)")
```

### Cell 2: Copy Data & Code

```python
import shutil, os
from pathlib import Path

# === TÌM DATASET TRONG /kaggle/input ===
CODE_INPUT = None
IMG_INPUT = None

for root_dir in Path("/kaggle/input").iterdir():
    if not root_dir.is_dir():
        continue
    # Tìm code: có thư mục src/holmhz
    for p in root_dir.rglob("holmhz"):
        if p.is_dir() and (p.parent.name == "src"):
            CODE_INPUT = p.parent.parent
            break
    # Tìm images: có thư mục rvf10k hoặc raw_v2
    for p in root_dir.rglob("rvf10k"):
        if p.is_dir():
            IMG_INPUT = p.parent
            break

if CODE_INPUT is None:
    print("❌ Code dataset not found!")
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
        print(f"  ⚠️ Not found: {src_dir}")

# Verify
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

fixed_total = 0
errors = []

for manifest_name in ["train.json", "val.json", "test_id.json", "test_ood.json"]:
    path = Path(f"data/manifests_v2/{manifest_name}")
    data = json.load(open(path))
    fixed = 0

    for entry in data:
        old_path = entry["path"]
        if "raw_v2" in old_path:
            idx = old_path.find("raw_v2")
            rel = old_path[idx:]
            rel = rel.replace("\\\\", "/").replace("\\", "/")
            new_path = "data/" + rel
            entry["path"] = new_path
            fixed += 1

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {manifest_name}: fixed {fixed}/{len(data)} paths")
    fixed_total += fixed

# Verify
print(f"\n  Verifying paths...")
train_data = json.load(open("data/manifests_v2/train.json"))

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

# Verify freq_fft is registered
assert "freq_fft" in DETECTOR_REGISTRY.list(), "freq_fft not registered!"
print("✅ freq_fft detector available!")

# Verify data counts
import json
train_data = json.load(open("data/manifests_v2/train.json"))
print(f"Train: {len(train_data)} samples")
fake = sum(1 for d in train_data if d["label"] == 1)
print(f"  Real: {len(train_data) - fake}, Fake: {fake} ({fake/len(train_data)*100:.1f}%)")

# Quick model test
import torch
model = DETECTOR_REGISTRY.build("freq_fft", dropout=0.3, use_phase=True)
x = torch.randn(2, 3, 224, 224)
logits = model(x)
print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Forward pass: {x.shape} → {logits.shape} ✅")
del model, x, logits
torch.cuda.empty_cache()
```

### Cell 5: Write v11 Config

```python
# ══════════════════════════════════════════════════
# FFT Frequency Detector — Custom 3-layer CNN
# Optimized for Kaggle T4 x2 (DataParallel)
# v11.1 FIX: LR 0.0003, batch 32/GPU, patience 10
# ══════════════════════════════════════════════════

import torch
n_gpus = torch.cuda.device_count()

# T4x2: batch 32/GPU → total 64; T4x1: batch 32
batch_size = 32 * max(1, n_gpus)
num_workers = 4 * max(1, n_gpus)

print(f"GPUs: {n_gpus}")
print(f"Batch size: {batch_size} ({32} × {max(1, n_gpus)} GPUs)")
print(f"Num workers: {num_workers}")

config_v11 = f"""
model:
  name: freq_fft
  dropout: 0.3
  use_phase: true

training:
  epochs: 40
  batch_size: {batch_size}
  learning_rate: 0.0003
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  pos_weight: 1.0
  early_stopping:
    patience: 10
    monitor: val_auc
  use_data_parallel: true

data:
  train_manifest: data/manifests_v2/train.json
  val_manifest: data/manifests_v2/val.json
  image_size: 224
  num_workers: {num_workers}
  augmentation: true
  use_weighted_sampler: true

wandb:
  project: holmhz
  entity: null
  log_every_n_steps: 10
"""

import os
os.makedirs("configs", exist_ok=True)
with open("configs/train_v11_freq.yaml", "w") as f:
    f.write(config_v11)
print(f"✅ configs/train_v11_freq.yaml written")
print(f"   model: freq_fft (FFT CNN ~1.7M params)")
print(f"   lr: 0.0003 (stable), batch: {batch_size}, epochs: 40, patience: 10")
```

### Cell 6: Train!

```python
import os
# Clean old checkpoints
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)

os.makedirs("outputs/checkpoints", exist_ok=True)

# ══════════════════════════════════════════════════
# Train FrequencyDetector (FFT CNN)
# Estimated: ~20-30 min on T4x2 (model nhỏ, 1.7M params)
# ══════════════════════════════════════════════════
print("Training FrequencyDetector (FFT CNN) v11...")
print(f"  DataParallel: {torch.cuda.device_count()} GPUs")
!PYTHONPATH=src python scripts/train.py configs/train_v11_freq.yaml data.num_workers=8
```

### Cell 7: Save + Evaluate

```python
import shutil, json, os
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_v11_freq.pt")
    size_mb = best.stat().st_size / 1e6
    print(f"✅ Saved /kaggle/working/best_v11_freq.pt ({size_mb:.1f}MB)")
else:
    print("❌ best.pt not found!")

# Write test config
test_config = """
model:
  name: freq_fft
  checkpoint: outputs/checkpoints/best.pt
  dropout: 0.3
  use_phase: true

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
  output_dir: outputs/evaluation_v11

wandb:
  project: holmhz
  log_results: false
"""
with open("configs/test_v11.yaml", "w") as f:
    f.write(test_config)
print("✅ configs/test_v11.yaml written")

# Full evaluation
print("\n=== Full Evaluation (ID + OOD) ===")
!PYTHONPATH=src python scripts/test.py configs/test_v11.yaml
```

---

## So sánh v11 vs v9

| Metric | v9 (EfficientNet) | v9 (CLIP) | v11 (FFT CNN) |
|--------|:-:|:-:|:-:|
| Trainable params | 4M | 769 | **1.7M** |
| Training time (T4x2) | ~60-90min | ~15-30min | **~20-30min** |
| VRAM per GPU | ~4GB | ~8GB | **~2-3GB** |
| Batch size (T4x2) | 32 | 32 | **128** (64/GPU) |
| Input | RGB 224×224 | RGB 224×224 | RGB → FFT spectrum |
| Purpose | Pixel-level patterns | Semantic understanding | **Frequency artifacts** |

## Tối ưu cho T4 x2

| Setting | v9 | v11 (Optimized) |
|---------|:--:|:--:|
| **batch_size** | 32 | **128** (64/GPU × 2) |
| **num_workers** | 4 | **8** (4/GPU × 2) |
| **DataParallel** | ❌ | **✅ auto** |
| **LR** | 0.0003 | **0.001** (model nhỏ hơn) |
| **Epochs** | 30 | **30** |
| **AMP (mixed precision)** | ✅ | ✅ |

> **Lý do batch lớn hơn**: FrequencyDetector chỉ 1.7M params (vs EfficientNet 4M, CLIP 304M).
> VRAM usage thấp hơn nhiều → batch_size 64/GPU dễ dàng trên T4 (16GB).

## Lưu ý quan trọng

1. **Cùng dataset v8/v9** — Không cần upload ảnh mới
2. **FFT tự động** — Model tự convert RGB → FFT trong forward pass, dùng cùng DataLoader
3. **Amplitude + Phase** — `use_phase: true` (2-channel input) tốt hơn chỉ amplitude
4. **Sau khi train**: Download `best_v11_freq.pt` → sẽ integrate vào Ensemble:
   ```
   p_final = (0.30 × EffNet + 0.40 × CLIP + 0.25 × FFT) × EXIF_multiplier
   ```
