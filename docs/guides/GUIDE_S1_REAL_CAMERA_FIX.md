# 📋 Strategy 1 — Fix real_camera: COCO 2017 Val + Retrain v5

> **Mục tiêu**: real_camera accuracy 36% → **≥ 60%**, OOD AUC 0.78 → **≥ 0.82**
> **Thời gian ước tính**: 3-5 giờ (download COCO ~15 phút, preprocess ~10 phút, train ~35 phút, evaluate ~5 phút)
> **Prerequisite**: v4 checkpoint đã có (OOD AUC 0.7838), threshold 0.76 đã áp dụng (S2)
> **Data source**: COCO 2017 Val Images — 5,000 ảnh real-world camera, cực đa dạng

---

## Mục lục

1. [Tại sao COCO 2017?](#1-tại-sao-coco-2017)
2. [Step 1 — Download COCO 2017 Val](#2-step-1--download-coco-2017-val)
3. [Step 2 — Subset 300 ảnh](#3-step-2--subset-300-ảnh)
4. [Step 3 — Resize + Rebuild Manifests](#4-step-3--resize--rebuild-manifests)
5. [Step 4 — Tạo Kaggle zip + upload](#5-step-4--tạo-kaggle-zip--upload)
6. [Step 5 — Train v5 trên Kaggle](#6-step-5--train-v5-trên-kaggle)
7. [Step 6 — Evaluate](#7-step-6--evaluate)
8. [Config v5 chi tiết](#8-config-v5-chi-tiết)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Tại sao COCO 2017?

### 1.1 Vấn đề hiện tại

Model v4 predict 64/100 ảnh `real_camera` là FAKE (P(Fake) median = 0.897).
Threshold tuning (S2) chỉ cứu thêm 6% → cần thêm training data.

### 1.2 Tại sao COCO là nguồn tối ưu nhất?

| Tiêu chí | COCO 2017 Val | Pexels/Unsplash | Google Images |
|---|---|---|---|
| **Số lượng** | 5,000 sẵn có | Phải download từng tấm | Phải download tay |
| **Đa dạng** | 80 categories (person, car, food, animal, landscape...) | Thiên landscape | Random |
| **Origin** | Camera thật (Flickr) | Stock photos | Không rõ |
| **License** | CC BY 4.0 (dùng được cho NCKH) | Pexels License | ⚠️ Không rõ |
| **Reproducible** | ✅ 1 link cố định | ❌ API thay đổi | ❌ Không |
| **1 file download** | ✅ val2017.zip (1GB) | ❌ Phải script | ❌ Tay |

### 1.3 80 categories của COCO (tất cả đều camera thật)

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, bench, bird, cat, dog, horse,
sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, bottle, wine glass, cup, fork, knife, spoon, bowl,
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
donut, cake, chair, couch, potted plant, bed, dining table, toilet,
tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear,
hair drier, toothbrush
```

→ **Cực kỳ đa dạng**: phong cảnh, đồ vật, động vật, thức ăn, phương tiện, nội thất, con người trong hoạt cảnh thật. Chính xác loại ảnh mà `real_camera` OOD đại diện.

---

## 2. Step 1 — Download COCO 2017 Val

### 2.1 Download

```bash
# Cách 1: Trình duyệt (nhanh nhất)
# Truy cập: http://images.cocodataset.org/zips/val2017.zip
# Download file ~1GB → lưu vào bất kỳ đâu

# Cách 2: Command line (PowerShell)
Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/val2017.zip" -OutFile "data/raw/val2017.zip"

# Cách 3: Command line (bash/curl)
curl -L -o data/raw/val2017.zip http://images.cocodataset.org/zips/val2017.zip
```

### 2.2 Extract

```bash
# Extract vào data/raw/coco_val2017/
# Windows: chuột phải → Extract All → data/raw/coco_val2017/
# Hoặc PowerShell:
Expand-Archive -Path "data/raw/val2017.zip" -DestinationPath "data/raw/coco_val2017"

# Sau khi extract, cấu trúc:
# data/raw/coco_val2017/val2017/
#   ├── 000000000139.jpg
#   ├── 000000000285.jpg
#   ├── ... (5,000 files)
```

> **Lưu ý**: COCO extract sẽ tạo subfolder `val2017/` bên trong. Path thực tế là `data/raw/coco_val2017/val2017/` — script ở Step 2 sẽ tự detect.

### 2.3 Verify

```bash
# Đếm số ảnh
python -c "from pathlib import Path; imgs=[f for f in Path('data/raw/coco_val2017').rglob('*.jpg')]; print(f'{len(imgs)} images found')"
# Expected: 5000 images found
```

---

## 3. Step 2 — Subset 300 ảnh

### 3.1 Tạo script

Tạo file `scripts/subset_coco_real.py`:

```python
"""
Subset 300 ảnh từ COCO 2017 Val → real_camera_train.

COCO Val chứa 5,000 ảnh camera thật, cực đa dạng (80 categories).
Lấy random 300 ảnh (seed=42) cho reproducibility.

Usage:
    python scripts/subset_coco_real.py

Output:
    data/raw/real/real_camera_train/  (300 images)
"""

import random
import shutil
from pathlib import Path

# === CONFIG ===
SEED = 42
SAMPLE_COUNT = 300
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# COCO extract path (auto-detect subfolder)
COCO_ROOT = Path("data/raw/coco_val2017")
DST_DIR = Path("data/raw/real/real_camera_train")


def find_coco_images(root: Path) -> list[Path]:
    """Tìm tất cả ảnh COCO, tự detect subfolder."""
    # COCO thường extract vào val2017/ subfolder
    candidates = [
        root / "val2017",        # data/raw/coco_val2017/val2017/
        root,                     # data/raw/coco_val2017/ (nếu extract trực tiếp)
    ]

    for folder in candidates:
        if folder.exists():
            images = sorted([
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ])
            if len(images) > 100:
                return images

    return []


def main():
    print("=" * 50)
    print("Subset COCO 2017 Val → real_camera_train")
    print("=" * 50)

    # 1. Tìm images
    images = find_coco_images(COCO_ROOT)
    if not images:
        print(f"❌ COCO images not found at: {COCO_ROOT}")
        print(f"   Download từ: http://images.cocodataset.org/zips/val2017.zip")
        print(f"   Extract vào: {COCO_ROOT}/")
        return

    print(f"✅ Found {len(images)} COCO Val images")

    # 2. Random subset
    random.seed(SEED)
    selected = random.sample(images, min(SAMPLE_COUNT, len(images)))
    print(f"📋 Selected {len(selected)} random images (seed={SEED})")

    # 3. Copy
    DST_DIR.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in DST_DIR.iterdir() if f.is_file()}

    copied, skipped = 0, 0
    for img in selected:
        if img.name in existing:
            skipped += 1
            continue
        shutil.copy2(img, DST_DIR / img.name)
        copied += 1

    print(f"\n✅ Copied: {copied} new, {skipped} skipped (already exist)")
    print(f"📁 Output: {DST_DIR}/ ({len(list(DST_DIR.iterdir()))} files)")

    # 4. Verify size diversity
    from PIL import Image
    sizes = set()
    for img_path in list(DST_DIR.iterdir())[:50]:
        try:
            w, h = Image.open(img_path).size
            sizes.add((w, h))
        except Exception:
            pass
    print(f"📐 Resolution diversity: {len(sizes)} unique sizes in first 50 images")
    print(f"\n📋 Next: python scripts/resize_all.py")


if __name__ == "__main__":
    main()
```

### 3.2 Chạy

```bash
python scripts/subset_coco_real.py
```

**Output mong đợi**:
```
✅ Found 5000 COCO Val images
📋 Selected 300 random images (seed=42)
✅ Copied: 300 new, 0 skipped
📁 Output: data/raw/real/real_camera_train/ (300 files)
📐 Resolution diversity: ~30 unique sizes in first 50 images
```

---

## 4. Step 3 — Resize + Rebuild Manifests

### 4.1 Cập nhật `scripts/resize_all.py`

Thêm **1 dòng** vào `folders_to_process` (sau dòng `real/real_pexels_train`, khoảng line 115):

```python
        # ── Training: Real (Task 1.7 v5 - COCO camera outdoor) ──
        ("real/real_camera_train",      "train/real/real_camera_train"),
```

### 4.2 Resize

```bash
python scripts/resize_all.py
```

**Output mong đợi**:
```
📁 real/real_camera_train (300 images) → train/real/real_camera_train
  real_camera_train: 100%|██████████| 300/300 [00:xx<00:00]
  ✅ 300 resized, 0 skipped, 0 errors
```

### 4.3 Rebuild manifests

```bash
python preprocessing/build_splits.py
```

> `build_splits.py` tự động scan `data/processed/train/real/` → sẽ tìm thấy `real_camera_train/` → thêm vào manifests. **Không cần sửa** `build_splits.py`.

**Output mong đợi**:
```
📂 Scanning training data...
  ✅ real/cifake: 7000 ảnh (label=0)
  ✅ real/diverse_real: 3000 ảnh (label=0)
  ✅ real/ffhq: 5000 ảnh (label=0)
  ✅ real/real_camera_train: 300 ảnh (label=0)      ← MỚI
  ✅ real/real_pexels_train: 300 ảnh (label=0)
  ✅ fake_gan/stylegan: 5000 ảnh (label=1)
  ✅ fake_diffusion/cifake: 7000 ảnh (label=1)
  ✅ fake_diffusion/sd15: 2500 ảnh (label=1)
  ✅ fake_diffusion/tristanzhang_train: 200 ảnh (label=1)

📊 Tổng cộng training data: 30300 ảnh
   Real: 15600   (trước: 15300)
   Fake: 14700

SPLIT RESULTS:
  Train     : 21210 ảnh   (trước: 21000)
  Val       :  4545 ảnh
  Test ID   :  4545 ảnh
  Test OOD  :   680 ảnh   (giữ nguyên)
```

### 4.4 Verify

```bash
python -c "
import json
with open('data/manifests/train.json') as f:
    d = json.load(f)
srcs = {}
for e in d:
    srcs[e['source']] = srcs.get(e['source'], 0) + 1
print(f'Total train: {len(d)}')
for s, c in sorted(srcs.items()):
    print(f'  {s:30s} {c}')
assert 'real_camera_train' in srcs, 'FAIL: real_camera_train missing!'
print(f'\n✅ PASS: real_camera_train = {srcs[\"real_camera_train\"]} samples')
"
```

**Output mong đợi**:
```
Total train: 21210
  cifake                          9800
  diverse_real                    2100
  ffhq                            3500
  real_camera_train                210   ← MỚI (300 × 0.7 = 210 train)
  real_pexels_train                210
  sd15                            1750
  stylegan                        3500
  tristanzhang_train               140

✅ PASS: real_camera_train = 210 samples
```

---

## 5. Step 4 — Tạo Kaggle zip + Upload

### 5.1 Tạo zip

```bash
python _create_kaggle_zip.py
```

`_create_kaggle_zip.py` scan `data/processed/train` recursively → tự động include `real_camera_train/`. **Không cần sửa**.

### 5.2 Verify zip chứa đủ data

```bash
python -c "
import zipfile
z = zipfile.ZipFile('holmhz-data-v3.zip')
cam = [n for n in z.namelist() if 'real_camera_train' in n]
tri = [n for n in z.namelist() if 'tristanzhang_train' in n]
print(f'real_camera_train: {len(cam)} files')
print(f'tristanzhang_train: {len(tri)} files')
assert len(cam) >= 290, f'FAIL: expected ~300, got {len(cam)}'
assert len(tri) >= 190, f'FAIL: expected ~200, got {len(tri)}'
print('✅ ZIP OK')
"
```

### 5.3 Upload lên Kaggle

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Tìm dataset hiện tại của bạn → **New Version**
3. Upload `holmhz-data-v3.zip` (~2.1 GB)
4. Đợi processing xong (~10-15 phút)

---

## 6. Step 5 — Train v5 trên Kaggle

Dùng **cùng template** từ `docs/KAGGLE_TRAINING_V4.md`, chỉ sửa **Cell 5** và **Cell 6-7**.

### Cell 1-4: Giữ nguyên từ v4

- Cell 1: pip install deps
- Cell 2: Auto-detect + copy data
- Cell 3: Tạo tristanzhang_train + rebuild manifests
- Cell 4: `sys.path.insert(0, "src")`

> **Lưu ý Cell 3**: Vẫn cần chạy Cell 3 (tristanzhang split) vì zip chỉ chứa processed data, không chứa filter files.

### Cell 5 — Config v5 (SỬA)

```python
# ═══ Cell 5: Write config v5 ═══
config_v5 = """
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
  pos_weight: 1.0
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

with open("configs/train_v5.yaml", "w") as f:
    f.write(config_v5)
print("✅ configs/train_v5.yaml written")
```

### Cell 6 — Train v5 (SỬA)

```python
# ═══ Cell 6: Train v5 ═══
import os, shutil, json

# Xóa old checkpoints
for f in ["outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"🗑️ Removed {f}")
os.makedirs("outputs/checkpoints", exist_ok=True)

# ─── Verify data ───
with open("data/manifests/train.json") as f:
    td = json.load(f)
sources = {}
for e in td:
    sources[e["source"]] = sources.get(e["source"], 0) + 1

print(f"\n📊 Train: {len(td)} samples")
for s, c in sorted(sources.items()):
    marker = " ← NEW" if s == "real_camera_train" else ""
    print(f"  {s:30s} {c:5d}{marker}")

# ─── Critical checks ───
ok = True
if "real_camera_train" not in sources:
    print("❌ FAIL: real_camera_train MISSING!")
    ok = False
if "tristanzhang_train" not in sources:
    print("❌ FAIL: tristanzhang_train MISSING!")
    ok = False
if len(td) < 21000:
    print(f"❌ FAIL: Expected ≥21000, got {len(td)}")
    ok = False

if not ok:
    raise RuntimeError("Data verification failed! Fix trước khi train.")

print(f"\n✅ ALL {len(td)} samples verified — Starting training...")

# ─── Train ───
!PYTHONPATH=src python scripts/train.py configs/train_v5.yaml data.num_workers=4
```

### Cell 7 — Save + Quick Eval (SỬA)

```python
# ═══ Cell 7: Save results ═══
import shutil
from pathlib import Path

# Copy checkpoint
best = Path("outputs/checkpoints/best.pt")
if best.exists():
    shutil.copy2(best, "/kaggle/working/best_v5.pt")
    size_mb = best.stat().st_size / 1e6
    print(f"✅ Saved /kaggle/working/best_v5.pt ({size_mb:.1f} MB)")
else:
    print("⚠️ best.pt not found!")

last = Path("outputs/checkpoints/last.pt")
if last.exists():
    shutil.copy2(last, "/kaggle/working/last_v5.pt")
    print("✅ Saved last_v5.pt")

# Quick OOD eval (threshold=0.76 từ S2)
print("\n" + "=" * 50)
print("QUICK OOD EVALUATION (threshold=0.76)")
print("=" * 50)
!PYTHONPATH=src python scripts/test.py model.checkpoint=outputs/checkpoints/best.pt \
    data.num_workers=4 data.batch_size=32 evaluation.threshold=0.76
```

---

## 7. Step 6 — Evaluate

### 7.1 Download checkpoint

1. Kaggle → Output tab → download `best_v5.pt`
2. Copy vào: `outputs/checkpoints/best_v5.pt`

### 7.2 Evaluate trên local

```bash
python scripts/test.py model.checkpoint=outputs/checkpoints/best_v5.pt data.num_workers=0 data.batch_size=32
```

> `configs/test.yaml` đã có `threshold: 0.76` (S2 đã áp dụng).

### 7.3 Re-run threshold analysis (tùy chọn)

```bash
python analysis/find_threshold.py model.checkpoint=outputs/checkpoints/best_v5.pt
```

### 7.4 Target metrics

| Metric | v4 (hiện tại) | v5 Target | Có thể giảm nhẹ? |
|---|---|---|---|
| ID AUC | 0.9972 | ≥ 0.99 | ❌ Không chấp nhận |
| OOD AUC | 0.7838 | ≥ **0.82** | - |
| **real_camera** | **36%** | ≥ **60%** | - |
| real_pexels | 74.5% | ≥ 70% | ✅ Có thể giảm nhẹ 5% |
| flux | 77.5% | ≥ 65% | ✅ Có thể giảm nhẹ |
| tristanzhang | 79.0% | ≥ 65% | ✅ Có thể giảm nhẹ |

### 7.5 Sau khi có kết quả → Decision

```
v5 OOD AUC ≥ 0.80?
├── YES → ✅ Close Task 1.7
└── NO  → OOD AUC vẫn ≥ 0.75?
    ├── YES → ✅ Close Task 1.7 (v4 target đã vượt)
    └── NO  → ⚠️ Regression — dùng best_v4.pt thay vì v5

real_camera ≥ 50%?
├── YES → ✅ Improvement! Dùng v5
└── NO  → ⚠️ Dùng v4 nếu OOD AUC tốt hơn, ghi nhận limitation

→ DÙ KẾT QUẢ NÀO → CLOSE TASK 1.7 → SPRINT 2
```

### 7.6 Cập nhật CONTEXT.md

Thêm section `17.22 v5 Results`:

```markdown
### 17.22 v5 Training Results (dd/mm/2026)

**Config**: train_v5.yaml — +300 COCO outdoor + pos_weight=1.0
**Changes vs v4**: +real_camera_train (COCO), pos_weight 1.2→1.0

| Metric | ID | OOD | Target |
|---|---|---|---|
| AUC | ? | ? | >0.82 |
| Acc | ? | ? | |

| OOD Source | Acc | Δ vs v4 |
|---|---|---|
| flux | ? | ? |
| tristanzhang_fake | ? | ? |
| real_pexels | ? | ? |
| real_camera | ? | ? |
```

---

## 8. Config v5 chi tiết

### 8.1 Thay đổi so với v4

| Param | v4 | **v5** | Lý do |
|---|---|---|---|
| **Training data** | 21,000 | **~21,210** | +300 COCO → 210 train (70% split) |
| **pos_weight** | 1.2 | **1.0** | Bớt FAKE bias → giúp real recognition |
| **threshold** (eval) | 0.5 | **0.76** | Youden J optimal (đã áp dụng S2) |
| sampler | true | true | Giữ — upsample minority sources |
| epochs | 30 | 30 | Giữ |
| lr | 1e-4 | 1e-4 | Giữ |
| patience | 10 | 10 | Giữ |

### 8.2 WeightedSampler effect (v5)

```
Source               | Raw N  | Effective/epoch | vs v4
────────────────────────────────────────────────────
cifake               | 9,800  | ~2,600          | giữ
ffhq                 | 3,500  | ~2,600          | giữ
stylegan             | 3,500  | ~2,600          | giữ
diverse_real          | 2,100  | ~2,600          | giữ
sd15                 | 1,750  | ~2,600          | giữ
real_pexels_train    | 210    | ~2,600          | giữ
tristanzhang_train   | 140    | ~2,600          | giữ
real_camera_train    | 210    | ~2,600   ← MỚI  | +2,600 effective real outdoor!
```

→ Model sẽ thấy **~2,600 real camera/outdoor images mỗi epoch** (trước: 0).

### 8.3 Tại sao 300 ảnh đủ?

- 300 raw → 210 train (70% split)
- WeightedSampler: 210 → **~2,600** effective/epoch (×12 upsample)
- Kết hợp augmentation (flip, crop, JPEG, blur) → mỗi epoch thấy 2,600 variations khác nhau
- `real_pexels_train` chỉ 210 ảnh nhưng đạt 74.5% → cùng logic

---

## 9. Troubleshooting

### 9.1 COCO download quá chậm

Mirror alternatives:
```bash
# Academic Torrents (nếu có torrent client)
# Hoặc Kaggle dataset: search "COCO 2017" trên Kaggle

# Hoặc dùng ít ảnh hơn: sửa SAMPLE_COUNT = 200 trong script
```

### 9.2 Không đủ dung lượng cho COCO full (1GB)

```bash
# Chỉ cần 300 ảnh (~50MB). Sau khi subset xong có thể xóa:
rm -rf data/raw/coco_val2017/
rm data/raw/val2017.zip
# Giữ lại: data/raw/real/real_camera_train/ (300 ảnh, ~50MB)
```

### 9.3 real_camera_train không có trong train.json

```bash
# Check processed folder
ls data/processed/train/real/real_camera_train/ | wc -l
# Phải = 300

# Nếu = 0 → resize chưa chạy:
python scripts/resize_all.py

# Nếu resize OK nhưng build_splits không thấy:
# Check xem folder name có đúng không (phải là real_camera_train, không phải real_camera)
```

### 9.4 Kaggle zip thiếu ảnh

```bash
# Verify
python -c "
import zipfile
z = zipfile.ZipFile('holmhz-data-v3.zip')
cam = [n for n in z.namelist() if 'real_camera_train' in n and n.endswith('.png')]
print(f'{len(cam)} real_camera_train images')
"
# Expected: 300
```

### 9.5 OOD AUC giảm sau v5 (regression)

Nếu OOD AUC < 0.75 (thấp hơn v4):
- **Nguyên nhân**: pos_weight=1.0 + thêm real → model bias REAL
- **Fix nhanh**: Dùng best_v4.pt (OOD AUC 0.78 đã đạt target)
- **Ghi báo cáo**: So sánh v4 vs v5, thảo luận trade-off

---

## Quick Reference — 10 Commands

```bash
# ═══ Pipeline hoàn chỉnh ═══

# 1. Download COCO Val 2017
curl -L -o data/raw/val2017.zip http://images.cocodataset.org/zips/val2017.zip

# 2. Extract
mkdir -p data/raw/coco_val2017 && unzip data/raw/val2017.zip -d data/raw/coco_val2017/

# 3. Subset 300 ảnh → real_camera_train
python scripts/subset_coco_real.py

# 4. Sửa resize_all.py (thêm 1 dòng mapping)
# → ("real/real_camera_train", "train/real/real_camera_train")

# 5. Resize
python scripts/resize_all.py

# 6. Rebuild manifests
python preprocessing/build_splits.py

# 7. Verify
python -c "import json; d=json.load(open('data/manifests/train.json')); print(len(d)); assert 'real_camera_train' in {e['source'] for e in d}"

# 8. Tạo zip
python _create_kaggle_zip.py

# 9. Upload Kaggle → Run notebook → Download best_v5.pt

# 10. Evaluate local
python scripts/test.py model.checkpoint=outputs/checkpoints/best_v5.pt data.num_workers=0 data.batch_size=32
```

---

*Guide version: S1-COCO — 03/03/2026. Dành riêng cho COCO 2017 Val data source.*
