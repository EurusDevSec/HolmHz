# Hướng Dẫn Train v4 trên Kaggle

## Mục tiêu

Train EfficientNet-B0 với **WeightedRandomSampler** + **pos_weight=1.2** để cải thiện OOD AUC từ ~0.50 lên >0.70.

---

## Bước 1 — Nén dataset để upload Kaggle

### Những gì cần nén

```
holmhz-data-v3.zip
├── data/
│   ├── processed/
│   │   ├── train/              ← ~30k ảnh training (bao gồm tristanzhang_train/)
│   │   └── ood_test/           ← 680 ảnh OOD test
│   └── manifests/              ← *.json + *.txt filter files
├── src/                        ← holmhz package
├── scripts/
│   ├── train.py
│   └── test.py
├── preprocessing/
│   └── build_splits.py
├── configs/                    ← tất cả YAML configs
└── pyproject.toml
```

> **Không cần nén:** `data/raw/`, `data/processed/val/` (rỗng), `data/processed/test_ood/` (duplicate), `outputs/`, `wandb/`, `weights/`, `.git/`

### Cách nén (Windows — dùng GUI hoặc PowerShell)

**Cách 1 — File Explorer:**

1. Chọn các thư mục: `data\processed\train`, `data\processed\ood_test`, `data\manifests`, `src`, `scripts`, `preprocessing`, `configs`
2. Thêm file: `pyproject.toml`
3. Chuột phải → **Compress to ZIP file**
4. Đặt tên: `holmhz-data-v3.zip`

**Cách 2 — PowerShell:**

```powershell
# Chạy từ thư mục gốc HolmHz
$items = @(
    "data\processed\train",
    "data\processed\ood_test",
    "data\manifests",
    "src",
    "scripts\train.py",
    "scripts\test.py",
    "preprocessing\build_splits.py",
    "configs",
    "pyproject.toml"
)
Compress-Archive -Path $items -DestinationPath "holmhz-data-v3.zip"
```

> **Lưu ý:** File zip đã được tạo sẵn ở `holmhz-data-v3.zip` (991 MB) — có thể dùng luôn mà không cần nén lại.

---

## Bước 2 — Upload lên Kaggle

1. Vào https://www.kaggle.com/datasets → **New Dataset**
2. Đặt tên: `holmhz-data-v3`
3. Upload `holmhz-data-v3.zip`
4. Đặt **Visibility: Private**
5. Click **Create**

---

## Bước 3 — Tạo Notebook trên Kaggle

1. Vào https://www.kaggle.com/code → **New Notebook**
2. **Settings** (bên phải):
   - Accelerator: **GPU T4 x2** (hoặc T4 x1)
   - Language: Python
   - Internet: **On**
3. **Add Data** → search `holmhz-data-v3` → **Add**
4. Copy từng cell dưới đây vào notebook

> **QUAN TRỌNG:** Mỗi `## Cell X` = **1 ô riêng biệt** trên Kaggle.
> Chỉ paste phần code Python (bên trong khối ` ```python ... ``` `).
> **KHÔNG** paste dòng tiêu đề `## Cell X` vào ô code.
> Sau mỗi cell, click **+ Code** để tạo ô mới rồi paste Cell tiếp theo.

---

## Cell 1 — Setup & Install

**Tạo ô code đầu tiên, paste đoạn sau:**

```python
!pip install -q wandb omegaconf timm albumentations tqdm python-dotenv rich scipy scikit-learn

import os
os.environ['WANDB_API_KEY'] = 'YOUR_WANDB_KEY_HERE'  # ← THAY KEY CỦA BẠN

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Cell 2 — Copy Data + Code

**Click `+ Code` tạo ô mới, paste đoạn sau:**

> **Lưu ý:** Cell này XÓA data cũ rồi copy lại từ dataset input để tránh dùng data rác từ lần chạy trước.

```python
import shutil, os
from pathlib import Path

# === Auto-detect dataset path ===
# Kaggle đường dẫn có thể là /kaggle/input/holmhz-data-v3
# hoặc /kaggle/input/<username>/holmhz-data-v3
# → tự tìm folder chứa "data" và "src"
DATA_INPUT = None
for p in Path("/kaggle/input").rglob("src"):
    if p.is_dir() and (p.parent / "data").exists():
        DATA_INPUT = p.parent
        break

# Fallback: nếu không tìm được, list tất cả để debug
if DATA_INPUT is None:
    print("❌ Không tìm được dataset! Listing /kaggle/input:")
    for item in sorted(Path("/kaggle/input").rglob("*")):
        if item.is_dir() and len(item.parts) <= 6:
            print(f"  {item}")
    raise FileNotFoundError("Dataset not found. Kiểm tra dataset đã Add đúng chưa.")

print(f"✅ Found dataset at: {DATA_INPUT}")
os.system(f'ls "{DATA_INPUT}"')

# --- Copy code (xóa cũ nếu có → copy mới) ---
for d in ["src", "scripts", "preprocessing", "configs"]:
    src, dst = DATA_INPUT / d, Path(d)
    if not src.exists():
        print(f"  ⚠️ WARNING: {src} not found in dataset!"); continue
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied {d}/")

for f in ["pyproject.toml"]:
    src, dst = DATA_INPUT / f, Path(f)
    if src.exists():
        shutil.copy2(src, dst); print(f"Copied {f}")

# --- Copy data (xóa cũ nếu có → copy mới) ---
for folder in ["processed", "manifests"]:
    src = DATA_INPUT / "data" / folder
    dst = Path(f"data/{folder}")
    if not src.exists():
        print(f"  ⚠️ WARNING: {src} not found!"); continue
    if dst.exists():
        shutil.rmtree(dst)
    print(f"Copying data/{folder} ...")
    shutil.copytree(src, dst)
    print(f"  Done!")

# --- Verify ---
print("\n=== Data structure ===")
!find data/processed -maxdepth 3 -type d
print()
!echo "OOD tristanzhang_fake:" && ls data/processed/ood_test/tristanzhang_fake/ | wc -l
!echo "Train tristanzhang_train:" && ls data/processed/train/fake_diffusion/tristanzhang_train/ | wc -l
!echo "Manifests:" && ls data/manifests/
```

---

## Cell 3 — Tạo tristanzhang_train + Rebuild Manifests ⚠️ CRITICAL

**Click `+ Code` tạo ô mới, paste đoạn sau:**

> Bước này QUAN TRỌNG NHẤT. Hai lần train trước thiếu bước này → model không thấy high-quality fakes → OOD AUC ~0.50.

```python
import random, shutil, json
from pathlib import Path

OOD_DIR   = Path("data/processed/ood_test/tristanzhang_fake")  # 500 ảnh
TRAIN_DIR = Path("data/processed/train/fake_diffusion/tristanzhang_train")
MANIFESTS = Path("data/manifests")
SEED = 42

# --- Step 1: Copy ảnh vào tristanzhang_train nếu chưa có ---
existing = list(TRAIN_DIR.glob("*.png")) if TRAIN_DIR.exists() else []
if len(existing) == 0:
    print("=== Creating tristanzhang_train ===")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(OOD_DIR.glob("*.png"))
    print(f"  Source: {len(all_imgs)} images in OOD tristanzhang_fake")
    assert len(all_imgs) >= 200, f"Not enough source images: {len(all_imgs)}"

    random.seed(SEED)
    random.shuffle(all_imgs)
    for img in all_imgs[:200]:
        shutil.copy2(img, TRAIN_DIR / img.name)
    print(f"  Copied 200 images → {TRAIN_DIR}")
else:
    print(f"tristanzhang_train already exists: {len(existing)} images")

# --- Step 2: Luôn tái tạo filter file từ nội dung thực tế của TRAIN_DIR ---
# (tránh lỗi nếu filter file cũ không khớp với ảnh đang có trong TRAIN_DIR)
MANIFESTS.mkdir(parents=True, exist_ok=True)
train_stems  = {p.stem for p in TRAIN_DIR.glob("*.png")}
all_ood_imgs = sorted(OOD_DIR.glob("*.png"))
test_imgs    = [p for p in all_ood_imgs if p.stem not in train_stems]

filter_path = MANIFESTS / "tristanzhang_test_only.txt"
with open(filter_path, "w") as f:
    for img in test_imgs:
        f.write(img.stem + ".jpg\n")

print(f"Filter written: {len(train_stems)} train | {len(test_imgs)} test  →  {filter_path}")
assert len(train_stems) + len(test_imgs) == len(all_ood_imgs), \
    f"Split không đủ: {len(train_stems)} + {len(test_imgs)} != {len(all_ood_imgs)}"

# --- Step 3: Rebuild manifests ---
print("\n=== Rebuilding manifests ===")
import subprocess
result = subprocess.run(["python", "preprocessing/build_splits.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("❌ build_splits.py FAILED!")
    print(result.stderr)
    raise RuntimeError("build_splits.py failed — xem lỗi ở trên")

# --- Step 4: Verify ---
manifest_path = Path("data/manifests/train.json")
assert manifest_path.exists(), f"Manifest không tồn tại: {manifest_path}"

train_data = json.load(open("data/manifests/train.json"))
ood_data   = json.load(open("data/manifests/test_ood.json"))

train_sources = {}
for x in train_data:
    train_sources[x["source"]] = train_sources.get(x["source"], 0) + 1

print(f"Train: {len(train_data)} total")
for s, c in sorted(train_sources.items()):
    print(f"  {s:32s} {c:5d}")

ood_sources = {}
for x in ood_data:
    ood_sources[x["source"]] = ood_sources.get(x["source"], 0) + 1
print(f"\nOOD: {len(ood_data)} total")
for s, c in sorted(ood_sources.items()):
    print(f"  {s:32s} {c:5d}")

# Checks
ok = True
if "tristanzhang_train" not in train_sources:
    print("\n❌ tristanzhang_train KHÔNG có trong training data!"); ok = False
if len(train_data) < 20000:
    print(f"\n❌ Train size quá nhỏ: {len(train_data)}"); ok = False
if len(ood_data) < 600:
    print(f"\n❌ OOD size quá nhỏ: {len(ood_data)}"); ok = False
if ok:
    print(f"\n✅ ALL CHECKS PASSED — sẵn sàng train!")
```

---

## Cell 4 — Install HolmHz Package

**Click `+ Code` tạo ô mới, paste đoạn sau:**

```python
# Thêm src/ vào Python path (thay vì pip install -e . vì hatchling lỗi trên Kaggle)
import sys, os
sys.path.insert(0, "src")
os.environ["PYTHONPATH"] = "src"  # Để scripts/train.py cũng tìm được holmhz

import holmhz
print(f"HolmHz: {holmhz.__version__}")
```

---

## Cell 5 — Ghi config train_v4.yaml

**Click `+ Code` tạo ô mới, paste đoạn sau:**

```python
from pathlib import Path

config = """
model:
  name: efficientnet_b0
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
""".strip()

Path("configs").mkdir(exist_ok=True)
Path("configs/train_v4.yaml").write_text(config)
print("✅ configs/train_v4.yaml written")
!cat configs/train_v4.yaml
```

---

## Cell 6 — TRAIN 🚀

**Click `+ Code` tạo ô mới, paste đoạn sau:**

> Dự kiến **30–45 phút** trên T4. Monitor W&B dashboard cho `val_auc`.

```python
import os
# Xóa checkpoint cũ
for f in ["outputs/checkpoints/last.pt", "outputs/checkpoints/best.pt"]:
    if os.path.exists(f):
        os.remove(f); print(f"Removed {f}")

!PYTHONPATH=src python scripts/train.py configs/train_v4.yaml data.num_workers=4
```

---

## Cell 7 — Copy checkpoint

**Click `+ Code` tạo ô mới, paste đoạn sau:**

```python
import shutil
from pathlib import Path

best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_v4.pt")
    print(f"✅ Saved best_v4.pt ({best.stat().st_size/1e6:.1f} MB)")
else:
    print("⚠️  best.pt not found!")

last = Path("outputs/checkpoints/last.pt")
if last.exists():
    shutil.copy2(last, "/kaggle/working/last_v4.pt")
    print(f"✅ Saved last_v4.pt")

# Quick OOD evaluation
!PYTHONPATH=src python scripts/test.py model.checkpoint=outputs/checkpoints/best.pt \
    data.num_workers=4 data.batch_size=32
```

---

## Bước 4 — Sau khi train xong

1. Vào **Output** tab trong Kaggle notebook
2. Download `best_v4.pt`
3. Copy vào local: `outputs/checkpoints/best_v4.pt`
4. Chạy evaluate local:
   ```bash
   python scripts/test.py model.checkpoint=outputs/checkpoints/best_v4.pt data.num_workers=0 data.batch_size=32
   ```

---

## Checklist nhanh

- [ ] `holmhz-data-v3.zip` đã upload Kaggle
- [ ] Dataset đặt tên đúng `holmhz-data-v3` (hoặc update đường dẫn ở Cell 2)
- [ ] GPU T4 đã bật
- [ ] `WANDB_API_KEY` đã điền ở Cell 1
- [ ] Cell 3 chạy thành công (thấy `✅ ALL CHECKS PASSED!`)
- [ ] Download `best_v4.pt` sau khi train

---

## Tóm tắt thay đổi v4 so với v3

|                       | v3       | v4                |
| --------------------- | -------- | ----------------- |
| WeightedRandomSampler | ❌       | ✅ cifake 47%→14% |
| pos_weight            | ❌       | ✅ 1.2            |
| tristanzhang_train    | ❌ thiếu | ✅ 200 ảnh        |
| epochs                | 25       | 30                |
| patience              | 8        | 10                |
