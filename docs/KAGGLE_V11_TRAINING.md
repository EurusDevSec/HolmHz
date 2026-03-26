# v11 FFT Frequency Detector — Kaggle Training Guide

## Tổng quan

Train `FrequencyDetector` (FFT CNN ~215K params) trên cùng dataset v8.
Model nhận ảnh RGB → chuyển sang phổ tần số (FFT) → phân loại Real/Fake.

**Cell structure giống v6/v9** (đã verified stable trên Kaggle).

---

## Cell 1: Setup & Install

```python
# Cell 1: Setup
!pip install -q wandb omegaconf albumentations timm grad-cam scipy rich

import os, sys, json, shutil
from pathlib import Path

# Kaggle paths
KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")

# Clone code
if not (WORK / "HolmHz").exists():
    !git clone https://github.com/YOUR_REPO/HolmHz.git {WORK}/HolmHz
    
os.chdir(str(WORK / "HolmHz"))
sys.path.insert(0, "src")

# Install holmhz
!pip install -e . --no-deps -q

print("✅ Setup done")
```

## Cell 2: Auto-discover datasets

```python
# Cell 2: Discover datasets
print("=== Kaggle Datasets ===")
for d in sorted(KAGGLE_INPUT.iterdir()):
    count = sum(1 for _ in d.rglob("*") if _.is_file())
    print(f"  {d.name}: {count} files")

# Symlink data
DATA_PROC = Path("data/processed")
if DATA_PROC.is_symlink():
    DATA_PROC.unlink()
elif DATA_PROC.exists():
    shutil.rmtree(DATA_PROC)

# Try auto-detect dataset  
for d in KAGGLE_INPUT.iterdir():
    candidate = d / "processed"
    if candidate.exists():
        DATA_PROC.parent.mkdir(parents=True, exist_ok=True)
        DATA_PROC.symlink_to(candidate)
        print(f"✅ Linked: {DATA_PROC} -> {candidate}")
        break
else:
    # Manual fallback
    DATA_PROC.parent.mkdir(parents=True, exist_ok=True)
    DATA_PROC.symlink_to(KAGGLE_INPUT / "holmhz-data-v8" / "processed")
    print(f"✅ Fallback linked")
```

## Cell 3: Fix manifest paths

```python
# Cell 3: Fix manifest paths for Kaggle
MANIFEST_DIR = Path("data/manifests")
for mf in MANIFEST_DIR.glob("*.json"):
    data = json.loads(mf.read_text())
    fixed = []
    for item in data:
        p = item["path"]
        # Replace Windows/local path with Kaggle path
        if "data/processed/" in p:
            p = str(DATA_PROC / p.split("data/processed/")[-1])
        item["path"] = p
        fixed.append(item)
    mf.write_text(json.dumps(fixed, indent=2))
    print(f"  Fixed {mf.name}: {len(fixed)} items")
    
print("✅ Manifests fixed")
```

## Cell 4: Write training config

```python
# Cell 4: Training config for FrequencyDetector
config_content = """
model:
  name: freq_fft
  dropout: 0.3
  use_phase: true

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  num_workers: 4
  
training:
  batch_size: 64
  epochs: 25
  lr: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  early_stopping:
    patience: 7
    min_delta: 0.001
  amp: true
  freeze_backbone: false

wandb:
  project: holmhz
  run_name: v11-freq-fft
"""

Path("configs/train_v11_freq.yaml").write_text(config_content)
print("✅ Config written")
```

## Cell 5: Train FrequencyDetector

```python
# Cell 5: Train
import torch
from omegaconf import OmegaConf
from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors

config = OmegaConf.load("configs/train_v11_freq.yaml")

# Build model
model = DETECTOR_REGISTRY.build(
    config.model.name,
    dropout=config.model.dropout,
    use_phase=config.model.use_phase,
)
print(f"Model: {model.__class__.__name__}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Train using existing trainer
from scripts.train import main as train_main
train_main("configs/train_v11_freq.yaml")
```

## Cell 6: Evaluate

```python
# Cell 6: Evaluate
!python scripts/test.py --config configs/test_v11.yaml
```

## Cell 7: Save checkpoint

```python
# Cell 7: Save
import shutil
best = Path("outputs/checkpoints/best.pt")
if best.exists():
    dst = WORK / "best_v11_freq.pt"
    shutil.copy(best, dst)
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"✅ Saved {dst} ({size_mb:.1f}MB)")
    
    # Save test config
    test_config = f"""
model:
  name: freq_fft
  dropout: 0.3
  use_phase: true
  checkpoint: outputs/checkpoints/best.pt
data:
  test_manifest: data/manifests/test_id.json
  ood_manifest: data/manifests/test_ood.json
  num_workers: 4
training:
  batch_size: 32
"""
    (WORK / "configs").mkdir(exist_ok=True)
    Path(WORK / "configs/test_v11.yaml").write_text(test_config)
    print("✅ configs/test_v11.yaml written")
```

## Lưu ý quan trọng

1. **Batch size 64** — FrequencyDetector chỉ 215K params, rất nhẹ → batch lớn OK
2. **LR 0.001** — Model nhỏ, cần LR cao hơn EfficientNet
3. **Epochs 25** — Train nhanh vì model nhỏ (~2-3 min/epoch trên T4)
4. **FFT tự động** — Model tự convert RGB → FFT trong forward pass, dùng cùng DataLoader v8
5. **Amplitude + Phase** — `use_phase=true` (2-channel input) → tốt hơn chỉ amplitude

## Sau khi train

Download `best_v11_freq.pt` về `outputs/checkpoints/` trên local.
Ensemble v11 sẽ tự load nếu có file.
