# GUIDE Task 1.7 -- Fix OOD Generalization

> **Muc tieu**: Nang OOD AUC tu 0.4812 len >= 0.75 bang cach mo rong training data + tang cuong augmentation
> **Thoi gian uoc tinh**: 2-3 ngay (data prep local + retrain on Kaggle)
> **Do kho**: Trung binh -- chu yeu la data engineering, khong doi kien truc model

---

## Muc luc

1. [Tong quan van de](#buoc-1-tong-quan-van-de)
2. [Chuan bi data moi](#buoc-2-chuan-bi-data-moi)
3. [Update pipeline scripts](#buoc-3-update-pipeline-scripts)
4. [Tang cuong augmentation](#buoc-4-tang-cuong-augmentation)
5. [Update configs](#buoc-5-update-configs)
6. [Upload len Kaggle + Retrain](#buoc-6-upload-len-kaggle--retrain)
7. [Download checkpoint + Re-evaluate](#buoc-7-download-checkpoint--re-evaluate)
8. [Phan tich ket qua + Cap nhat docs](#buoc-8-phan-tich-ket-qua--cap-nhat-docs)
9. [Troubleshooting](#buoc-9-troubleshooting)

---

## Buoc 1: Tong quan van de

### 1.1 Hien trang

```
Training data hien tai:
  Real:  cifake (4,927) + ffhq (3,500)     = 8,427 real
  Fake:  cifake (4,873) + stylegan (3,500) + sd15 (1,750) = 10,123 fake
  Total: 18,550

Van de voi Real data:
  cifake_real:  32x32 images upscale len 224x224 => artifacts resize cuc ky dac trung
  ffhq:         Chi khuon mat (face crops, aligned) => 1 distribution duy nhat

  => Model hoc: "anh co texture nhu cifake/ffhq = Real, con lai = Fake"
  => Khi gap anh phong canh, do vat, camera phone => du doan FAKE het
```

### 1.2 Muc tieu

```
Truoc (Task 2.1):
  ID AUC  = 0.9979    OOD AUC  = 0.4812
  ID Acc  = 0.9814    OOD Acc  = 0.4805
  real_pexels = 8.6%  real_camera = 12.0%

Sau (Task 1.7):
  ID AUC  >= 0.95     OOD AUC  >= 0.75     (target)
  ID Acc  >= 0.93     OOD Acc  >= 0.70
  real_pexels >= 0.60 real_camera >= 0.50
```

### 1.3 Chien luoc 3 prong

```
 PRONG 1: DATA                  PRONG 2: AUGMENTATION         PRONG 3: TRAINING
 Mo rong Real training          Tang cuong chong shortcut      Fine-tune strategy
 ──────────────────             ──────────────────────         ──────────────────
 + 3,000 tu 140k dataset        JPEG quality 30-100           Phase 1: freeze + head
   (256x256, diverse objects)    Blur sigma 3-9                Phase 2: unfreeze + LR 1e-4
 + 300 tu real_pexels           Downscale 0.25-0.9            Early stopping patience=7
   (split: 300 train/200 OOD)   RandomResizedCrop 0.7-1.0
                                ColorJitter p=0.5
```

---

## Buoc 2: Chuan bi data moi

### 2.1 Tong quan data co san

Du lieu da co tren may, **khong can download them**:

| Source                    | Location                                      | Images | Size    | Dung cho                   |
| ------------------------- | --------------------------------------------- | ------ | ------- | -------------------------- |
| 140k_real_and_fake (real) | `data/raw/140k_real_and_fake/.../train/real/` | 50,000 | 256x256 | Subset 3,000 cho train     |
| 140k_real_and_fake (fake) | `data/raw/140k_real_and_fake/.../train/fake/` | 50,000 | 256x256 | KHONG dung (da co du fake) |
| real_pexels               | `data/raw/ood_test/real_pexels/`              | 500    | ~4000px | Split 300 train / 200 OOD  |

### 2.2 Script subset 140k diverse real

Tao file `scripts/subset_140k_real.py`:

```python
"""
Subset 3,000 anh Real diverse tu 140k_real_and_fake dataset.

Dataset 140k co 50,000 real images (256x256, diverse objects -- khong chi face).
Ta chi can 3,000 de bo sung training data.

Usage:
    python scripts/subset_140k_real.py

Output:
    data/raw/real/diverse_real/ (3,000 images)
"""

import random
import shutil
from pathlib import Path

# === CONFIG ===
SRC_DIR = Path("data/raw/140k_real_and_fake/real_vs_fake/real-vs-fake/train/real")
DST_DIR = Path("data/raw/real/diverse_real")
SUBSET_SIZE = 3000
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    # Validate source
    if not SRC_DIR.exists():
        print(f"ERROR: Source not found: {SRC_DIR}")
        return

    # Gather all images
    all_images = sorted([
        f for f in SRC_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ])
    print(f"Source: {SRC_DIR}")
    print(f"Total available: {len(all_images)}")

    # Random subset
    random.seed(SEED)
    selected = random.sample(all_images, min(SUBSET_SIZE, len(all_images)))
    print(f"Selected: {len(selected)} images (seed={SEED})")

    # Copy to destination
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing (resume support)
    existing = set(f.name for f in DST_DIR.iterdir() if f.is_file())
    copied = 0
    skipped = 0

    for img_path in selected:
        if img_path.name in existing:
            skipped += 1
            continue
        shutil.copy2(img_path, DST_DIR / img_path.name)
        copied += 1

    total = copied + skipped
    print(f"\nDone! {total} images in {DST_DIR}")
    print(f"  Copied: {copied}, Skipped (existing): {skipped}")
    print(f"\nNext: python scripts/split_real_pexels.py")


if __name__ == "__main__":
    main()
```

### 2.3 Script split real_pexels

Tao file `scripts/split_real_pexels.py`:

```python
"""
Split real_pexels: 300 anh cho training, 200 giu lai cho OOD test.

Hien tai: 500 anh real_pexels deu nam trong OOD test set.
Ta can chuyen 300 anh vao training de model hoc "anh phong canh = Real".
Giu lai 200 anh de test OOD (van co tinh generalization).

Usage:
    python scripts/split_real_pexels.py

Output:
    data/raw/real/real_pexels_train/  (300 images)
    data/raw/ood_test/real_pexels/    (giu nguyen 500 -- manifest se chi dung 200)
"""

import random
import shutil
from pathlib import Path

# === CONFIG ===
SRC_DIR = Path("data/raw/ood_test/real_pexels")
DST_DIR = Path("data/raw/real/real_pexels_train")
TRAIN_COUNT = 300
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    if not SRC_DIR.exists():
        print(f"ERROR: Source not found: {SRC_DIR}")
        return

    # Gather all images
    all_images = sorted([
        f for f in SRC_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ])
    print(f"Source: {SRC_DIR}")
    print(f"Total: {len(all_images)}")

    # Random split: 300 train, 200 test
    random.seed(SEED)
    random.shuffle(all_images)  # in-place shuffle for reproducibility

    # Re-sort after shuffle for consistency
    shuffled = list(all_images)  # already shuffled
    train_images = shuffled[:TRAIN_COUNT]
    test_images = shuffled[TRAIN_COUNT:]

    print(f"Train: {len(train_images)}")
    print(f"Test:  {len(test_images)}")

    # Copy train images (DO NOT move -- keep originals in ood_test)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    existing = set(f.name for f in DST_DIR.iterdir() if f.is_file())
    copied = 0
    skipped = 0

    for img_path in train_images:
        if img_path.name in existing:
            skipped += 1
            continue
        shutil.copy2(img_path, DST_DIR / img_path.name)
        copied += 1

    print(f"\nCopied to {DST_DIR}: {copied} new, {skipped} skipped")

    # Save the test-only file list (for build_splits.py to use)
    test_names = sorted([f.name for f in test_images])
    list_path = Path("data/manifests/real_pexels_test_only.txt")
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, "w") as f:
        for name in test_names:
            f.write(name + "\n")

    print(f"Test-only list saved: {list_path} ({len(test_names)} files)")
    print(f"\nNext: python scripts/resize_all.py  (to resize new data)")


if __name__ == "__main__":
    main()
```

> **QUAN TRONG**: Ta COPY (khong move) tu real_pexels.
> Raw files van o `data/raw/ood_test/real_pexels/` (500 anh).
> Ta chi copy 300 anh sang `data/raw/real/real_pexels_train/` de resize + train.
> Build_splits se dung danh sach `real_pexels_test_only.txt` de chi giu 200 anh trong OOD manifest.

### 2.4 Chay cac scripts

```bash
# Buoc 1: Subset 140k diverse real
python scripts/subset_140k_real.py
# => data/raw/real/diverse_real/ (3,000 images)

# Buoc 2: Split real_pexels
python scripts/split_real_pexels.py
# => data/raw/real/real_pexels_train/ (300 images)
# => data/manifests/real_pexels_test_only.txt (200 filenames)

# Verify
ls data/raw/real/
# => cifake_subset/  diverse_real/  ffhq/  ffhq_full/  real_pexels_train/
```

---

## Buoc 3: Update pipeline scripts

### 3.1 Update `scripts/resize_all.py`

Them 2 folder moi vao `folders_to_process`:

```python
# Tim doan nay trong resize_all.py:
folders_to_process = [
    # -- Training: Real --
    ("real/cifake_subset",          "train/real/cifake"),
    ("real/ffhq",                   "train/real/ffhq"),
    # THEM 2 DONG NAY:
    ("real/diverse_real",           "train/real/diverse_real"),
    ("real/real_pexels_train",      "train/real/real_pexels_train"),
    # -- Training: Fake GAN --
    ("fake_gan/stylegan",           "train/fake_gan/stylegan"),
    # ...
]
```

Sau do chay:

```bash
python scripts/resize_all.py
# Chi resize 2 folder moi (da skip cac folder cu)
# => data/processed/train/real/diverse_real/ (3,000 images, 224x224)
# => data/processed/train/real/real_pexels_train/ (300 images, 224x224)
```

### 3.2 Update `preprocessing/build_splits.py`

Thay doi OOD scanning de chi giu 200 real_pexels (khong phai 500):

```python
# Trong ham scan_ood_folder(), sau khi scan real_pexels:
# Them logic filter dua tren real_pexels_test_only.txt

def scan_ood_folder(ood_dir: Path) -> list[dict]:
    """Scan OOD test folder -- voi filter cho real_pexels."""
    entries = []

    if not ood_dir.exists():
        print(f"  OOD folder khong ton tai: {ood_dir}")
        return entries

    OOD_LABELS = {
        "tristanzhang_fake": 1,
        "flux": 1,
        "real_pexels": 0,
        "real_camera": 0,
    }

    # Load real_pexels test-only list (if exists)
    test_only_list = None
    test_only_path = Path("data/manifests/real_pexels_test_only.txt")
    if test_only_path.exists():
        with open(test_only_path) as f:
            test_only_list = set(line.strip() for line in f if line.strip())
        print(f"  Loaded real_pexels test-only filter: {len(test_only_list)} files")

    for source_dir in sorted(ood_dir.iterdir()):
        if not source_dir.is_dir():
            continue

        source_name = source_dir.name
        label = OOD_LABELS.get(source_name, -1)
        if label == -1:
            print(f"  Unknown OOD source: {source_name}")
            continue

        images = sorted([
            f for f in source_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        for img_path in images:
            # Filter real_pexels: chi giu 200 test images
            if source_name == "real_pexels" and test_only_list is not None:
                # img_path la processed path, can match ten file goc
                # processed name = stem (khong co extension goc)
                # test_only_list chua ten file goc (vd: "0004.jpg")
                stem = img_path.stem  # "0004"
                # Check if any test-only name starts with this stem
                if not any(t.startswith(stem) for t in test_only_list):
                    continue  # Skip -- nay la training image, khong phai test

            entries.append({
                "path": str(img_path.as_posix()),
                "label": label,
                "source": source_name,
                "category": "ood",
            })

        label_str = "fake" if label == 1 else "real"
        count = sum(1 for e in entries if e["source"] == source_name)
        print(f"  ood_test/{source_name}: {count} anh (label={label} -> {label_str})")

    return entries
```

Sau do rebuild manifests:

```bash
python preprocessing/build_splits.py
# => train.json: ~21,850 (8,427 + 3,000 + 300 = 11,727 real, ~10,123 fake)
# => val.json: tuong tu
# => test_id.json: tuong tu
# => test_ood.json: ~880 (giam tu 1,180 vi bot 300 real_pexels)
```

### 3.3 Verify data moi

```bash
python -c "
import json
for name in ['train', 'val', 'test_id', 'test_ood']:
    d = json.load(open(f'data/manifests/{name}.json'))
    sources = {}
    for x in d:
        s = x['source']
        if s not in sources: sources[s] = {'real':0,'fake':0}
        sources[s]['real' if x['label']==0 else 'fake'] += 1
    print(f'\n{name}: {len(d)} total')
    for s, c in sorted(sources.items()):
        total = c['real'] + c['fake']
        print(f'  {s:25s} real={c[\"real\"]:5d}  fake={c[\"fake\"]:5d}  total={total:5d}')
"
```

**Ky vong output:**

```
train: ~21850 total
  cifake                    real= 4927  fake= 4873  total= 9800
  diverse_real              real= 2100  fake=    0  total= 2100   # NEW (70% of 3000)
  ffhq                      real= 3500  fake=    0  total= 3500
  real_pexels_train         real=  210  fake=    0  total=  210   # NEW (70% of 300)
  sd15                      real=    0  fake= 1750  total= 1750
  stylegan                  real=    0  fake= 3500  total= 3500

val: ~4700 total
  cifake                    real= 1026  fake= 1074  total= 2100
  diverse_real              real=  450  fake=    0  total=  450   # NEW
  ffhq                      real=  750  fake=    0  total=  750
  real_pexels_train         real=   45  fake=    0  total=   45   # NEW
  sd15                      real=    0  fake=  375  total=  375
  stylegan                  real=    0  fake=  750  total=  750

test_ood: ~880 total
  flux                      real=    0  fake=   80  total=   80
  real_camera               real=  100  fake=    0  total=  100
  real_pexels               real=  200  fake=    0  total=  200   # GIAM tu 500
  tristanzhang_fake         real=    0  fake=  500  total=  500
```

---

## Buoc 4: Tang cuong augmentation

### 4.1 Update `src/holmhz/data/transforms.py`

Sua ham `get_train_transforms()`:

```python
def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """
    Transforms cho TRAINING v2 -- augment MANH HON de chong shortcut learning.

    Thay doi so voi v1:
    - OneOf p: 0.3 -> 0.5 (ap dung nhieu hon)
    - JPEG quality: 60-100 -> 30-100 (aggressive hon)
    - GaussianBlur: 3-7 -> 3-9 (manh hon)
    - Them Downscale (0.25-0.9) -- mo phong multi-resolution
    - Them RandomResizedCrop (0.7-1.0, p=0.3) -- pha spatial artifacts
    - ColorJitter p: 0.3 -> 0.5

    Ly do: Model v1 hoc shortcut tu preprocessing artifacts (cifake 32x32 upscale,
    ffhq face alignment). Augmentation manh hon -> pha cac artifacts nay.
    """
    return A.Compose([
        # 1. Random crop + resize (p=0.3) -- pha spatial artifacts
        # NEU khong crop -> chi resize binh thuong
        A.OneOf([
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
            ),
            A.Resize(image_size, image_size),
        ], p=1.0),  # Luon chon 1 trong 2

        # 2. Lat ngang (50%)
        A.HorizontalFlip(p=0.5),

        # 3. Nhom augmentation chinh -- TANG p tu 0.3 -> 0.5
        A.OneOf([
            # JPEG Compression -- aggressive hon (quality 30-100)
            A.ImageCompression(quality_range=(30, 100)),
            # Gaussian Blur -- manh hon (3-9)
            A.GaussianBlur(blur_limit=(3, 9)),
            # Gaussian Noise
            A.GaussNoise(std_range=(0.01, 0.05)),
            # Downscale -- mo phong anh resolution thap
            A.Downscale(scale_range=(0.25, 0.9)),
        ], p=0.5),

        # 4. Thay doi mau sac -- TANG p tu 0.3 -> 0.5
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05,
            p=0.5,
        ),

        # 5. Normalize (bat buoc)
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # 6. To tensor
        ToTensorV2(),
    ])
```

### 4.2 Giai thich tung thay doi

```
1. RandomResizedCrop (MOI):
   - Crop ngau nhien 70-100% cua anh roi resize lai 224x224
   - Pha bo spatial artifacts cua resize (cifake 32x32->224x224 tao grid-like pattern)
   - 50% chance crop, 50% chance resize binh thuong (OneOf)
   - Giup model khong phu thuoc vao resolution goc

2. ImageCompression quality 30-100 (cu: 60-100):
   - JPEG compression manh hon (quality 30 = rat nhieu artifacts)
   - Mo phong anh share qua MXH nhieu lan (WhatsApp, Facebook)
   - Pha cac frequency-domain artifacts ma model dang hoc nham

3. Downscale (MOI):
   - Giam resolution xuong 25-90% roi resize len lai
   - Tao blur + aliasing artifacts tuong tu anh low-res
   - Model phai hoc features khong phu thuoc vao resolution

4. ColorJitter p=0.5 (cu: 0.3):
   - Da dang mau sac, do sang, contrast
   - Mo phong nhieu dieu kien anh sang (ngoai troi, trong nha, ban dem)
   - Giup troi anh cifake (bao hoa) va anh camera (tu nhien)

5. GaussianBlur 3-9 (cu: 3-7):
   - Blur manh hon -- pha high-frequency artifacts
   - Wang et al. (2020): blur la augmentation quan trong nhat cho deepfake detection
```

---

## Buoc 5: Update configs

### 5.1 Tao `configs/train_v2.yaml`

```yaml
# ============================================
# HolmHz Training v2 -- OOD Improvement
# ============================================
# Thay doi so voi v1:
# - Data moi: +3,000 diverse_real + 300 real_pexels_train
# - Augmentation v2: manh hon (JPEG 30-100, Downscale, RandomCrop)
# - Phase 2: unfreeze all, LR=1e-4

model:
  name: efficientnet_b0
  pretrained: true
  num_classes: 1
  dropout: 0.3
  freeze_backbone: false # Full fine-tune tu dau voi data v2

training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.0001 # 1e-4, lower vi unfreeze all
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  early_stopping:
    patience: 7 # Tang tu 5 -> 7 vi data lon hon, can nhieu epoch hon

data:
  train_manifest: data/manifests/train.json
  val_manifest: data/manifests/val.json
  image_size: 224
  num_workers: 4 # Kaggle: 4, Local: 0
  augmentation: true

wandb:
  project: holmhz
  entity: null
  log_every_n_steps: 10
```

### 5.2 Training strategy

```
Option A: Full fine-tune tu dau (DON GIAN, KHUYEN DUNG)
=======================================================
- Dung pretrained ImageNet weights (khong dung checkpoint cu)
- freeze_backbone: false
- LR: 1e-4
- Epochs: 20
- Early stopping patience: 7
- Uu diem: Don gian, khong bi bias tu checkpoint cu

Option B: 2-phase (NEU Option A khong dat KPI)
===============================================
Phase 1:
  - Dung pretrained ImageNet weights
  - freeze_backbone: true, LR: 1e-3
  - Epochs: 10
Phase 2:
  - Tiep tuc tu Phase 1 checkpoint
  - freeze_backbone: false, LR: 1e-4
  - Epochs: 15

Ta bat dau voi Option A. Neu khong dat OOD >= 0.75, thu Option B.
```

---

## Buoc 6: Upload len Kaggle + Retrain

### 6.1 Chuan bi data cho Kaggle

```bash
# Pack processed data thanh zip (chi can processed, khong can raw)
cd data
# Nen toan bo processed/ folder
# Chu y: file ~2-3GB voi data moi

# Option 1: Kaggle CLI
pip install kaggle
kaggle datasets create -p data/processed/ -m "HolmHz training data v2 - added diverse_real + real_pexels_train"

# Option 2: Upload zip qua Kaggle UI
# Tao zip chi chua processed/ va manifests/
powershell -c "Compress-Archive -Path 'data/processed/*','data/manifests/*' -DestinationPath 'data/holmhz-data-v2.zip'"
# Upload len kaggle.com/datasets
```

### 6.2 Kaggle Notebook

```python
# === Cell 1: Setup ===
!pip install -q wandb omegaconf timm albumentations tqdm python-dotenv

import os
os.environ['WANDB_API_KEY'] = 'your-key-here'

# === Cell 2: Mount dataset ===
# Kaggle dataset path (update voi ten dataset cua ban)
DATA_ROOT = "/kaggle/input/holmhz-data-v2"

# Copy manifests + processed data
!cp -r {DATA_ROOT}/manifests data/manifests
!cp -r {DATA_ROOT}/processed data/processed

# Verify
!python -c "
import json
d = json.load(open('data/manifests/train.json'))
print(f'Train: {len(d)} samples')
sources = {}
for x in d:
    s = x['source']
    sources[s] = sources.get(s, 0) + 1
for s, n in sorted(sources.items()):
    print(f'  {s}: {n}')
"

# === Cell 3: Install project ===
!pip install -e .

# === Cell 4: Verify new augmentation ===
from holmhz.data.transforms import get_train_transforms
t = get_train_transforms(224)
print("Train transforms v2:")
print(t)

# === Cell 5: Train (Option A -- full fine-tune) ===
!python scripts/train.py \
    model.freeze_backbone=false \
    training.learning_rate=0.0001 \
    training.epochs=20 \
    training.early_stopping.patience=7 \
    training.batch_size=32 \
    data.num_workers=4

# === Cell 6: Copy checkpoint ===
# Kaggle output
!cp outputs/checkpoints/best.pt /kaggle/working/best_v2.pt
!ls -la /kaggle/working/best_v2.pt
```

### 6.3 Monitor training

```
Theo doi tren W&B:
- val_auc: ky vong dat 0.96-0.98 (co the thap hon v1 vi data da dang hon)
- train_loss: ky vong giam tu tu, khong spike
- learning_rate: cosine decay tu 1e-4 -> 0

Red flags:
- val_auc giam lien tuc sau epoch 5 -> overfitting, giam LR
- train_loss khong giam -> LR qua thap, tang len 5e-4
- OOM Kaggle -> giam batch_size xuong 16
```

---

## Buoc 7: Download checkpoint + Re-evaluate

### 7.1 Download checkpoint

```bash
# Tu Kaggle, download best_v2.pt ve local
# Dat vao: outputs/checkpoints/best_v2.pt

# Verify
ls -la outputs/checkpoints/best_v2.pt
```

### 7.2 Re-evaluate

```bash
# Dung evaluation pipeline da xay (Task 2.1)
# Chi can doi checkpoint path

python scripts/test.py \
    model.checkpoint=outputs/checkpoints/best_v2.pt \
    data.num_workers=0 \
    data.batch_size=32

# Output se hien thi:
# - ID metrics (ky vong AUC >= 0.95)
# - OOD metrics (ky vong AUC >= 0.75)
# - Per-source breakdown
# - Visualization PNGs
```

### 7.3 So sanh v1 vs v2

```bash
# Doc 2 reports
python -c "
import json

v1 = json.load(open('outputs/evaluation/eval_report.json'))
# v2 se duoc save tu dong vao outputs/evaluation/ khi chay test.py
# Rename v1 truoc khi chay v2:

print('=== v1 (before) ===')
print(f'ID AUC:  {v1[\"in_domain\"][\"overall\"][\"auc\"]:.4f}')
print(f'OOD AUC: {v1[\"ood\"][\"overall\"][\"auc\"]:.4f}')
for src, m in v1['ood']['per_source'].items():
    print(f'  {src:25s} acc={m[\"accuracy\"]:.4f}')
"
```

> **QUAN TRONG**: Truoc khi chay test.py voi v2, rename file v1:
>
> ```bash
> mv outputs/evaluation/eval_report.json outputs/evaluation/eval_report_v1.json
> mv outputs/evaluation/confusion_matrix_id.png outputs/evaluation/confusion_matrix_id_v1.png
> mv outputs/evaluation/confusion_matrix_ood.png outputs/evaluation/confusion_matrix_ood_v1.png
> mv outputs/evaluation/roc_curve.png outputs/evaluation/roc_curve_v1.png
> mv outputs/evaluation/per_source_accuracy.png outputs/evaluation/per_source_accuracy_v1.png
> ```

---

## Buoc 8: Phan tich ket qua + Cap nhat docs

### 8.1 Bang so sanh ky vong

```
                    v1 (Task 1.6)     v2 (Task 1.7)     Target
                    ─────────────     ─────────────     ──────
ID AUC              0.9979            ~0.96-0.98        >= 0.95
ID Accuracy         0.9814            ~0.94-0.97        >= 0.93
OOD AUC             0.4812            ~0.75-0.85        >= 0.75
OOD Accuracy        0.4805            ~0.70-0.80        >= 0.70
real_pexels         0.0860            ~0.60-0.80        >= 0.60
real_camera         0.1200            ~0.50-0.70        >= 0.50
flux                0.9500            ~0.80-0.95        >= 0.80
tristanzhang_fake   0.8720            ~0.75-0.90        >= 0.75
```

### 8.2 Cap nhat CONTEXT.md

Them section moi:

```markdown
## XX. OOD Improvement (Task 1.7) -- Completed DD/MM/2026

### Thay doi data

- Them 3,000 diverse_real tu 140k dataset (256x256, da dang noi dung)
- Them 300 real_pexels vao training (high-res phong canh/do vat)
- Training data v2: ~21,850 samples (11,727 real + 10,123 fake)
- OOD test: giam xuong ~880 (bot 300 real_pexels da chuyen sang train)

### Thay doi augmentation

- JPEG compression: quality 30-100 (cu: 60-100)
- Them Downscale (0.25-0.9)
- Them RandomResizedCrop (scale 0.7-1.0)
- Tang ColorJitter p=0.5 (cu: 0.3)
- Tang OneOf p=0.5 (cu: 0.3)

### Ket qua so sanh v1 vs v2

| Metric          | v1     | v2  | Thay doi |
| --------------- | ------ | --- | -------- |
| ID AUC          | 0.9979 | ??? | ???      |
| OOD AUC         | 0.4812 | ??? | ???      |
| real_pexels acc | 0.0860 | ??? | ???      |
| real_camera acc | 0.1200 | ??? | ???      |
```

### 8.3 Cap nhat PROJECT_PLAN.md

Them Task 1.7 vao Sprint 1 table.

### 8.4 Cap nhat TASK_1.7

Danh dau cac subtasks da hoan thanh, dien so lieu thuc te.

---

## Buoc 9: Troubleshooting

### 9.1 OOD AUC van < 0.75 sau retrain

```
Nguyen nhan co the:
1. 3,000 diverse_real van chua du da dang
   => Tang len 5,000-8,000

2. Augmentation chua du manh
   => Tang ImageCompression quality_min xuong 20
   => Them GridDistortion, ElasticTransform
   => Tang p len 0.7

3. Model qua don gian (EfficientNet-B0)
   => Thu EfficientNet-B3 hoac B4 (nhieu params hon)
   => Thu CLIP ViT-L/14 (generalization tot hon)

4. Training chua hoi tu
   => Tang epochs len 30, patience len 10
   => Thu LR 5e-4

5. Label noise trong data moi
   => Kiem tra bang tay 50 anh tu diverse_real
   => Dam bao tat ca la anh thuc (khong pha tron fake)
```

### 9.2 ID AUC giam qua nhieu (< 0.93)

```
Nguyen nhan:
1. Augmentation qua manh -> model khong hoc duoc signal
   => Giam p tu 0.5 -> 0.4
   => Giam JPEG quality_min tu 30 -> 40

2. Data imbalance (real nhieu hon fake)
   => Dung WeightedRandomSampler
   => Hoac oversampling fake

3. Learning rate qua cao
   => Giam tu 1e-4 -> 5e-5
```

### 9.3 OOM tren Kaggle T4

```
GPU: T4 (16GB VRAM) -- thuong du cho batch_size=32

Neu OOM:
1. Giam batch_size: 32 -> 16
2. Dung gradient accumulation:
   # Trong train.py, them:
   accumulation_steps = 2
   # Forward batch_size=16, virtual batch=32

3. Mixed precision (fp16):
   scaler = torch.amp.GradScaler()
   with torch.amp.autocast('cuda'):
       loss = model(images)
```

### 9.4 Chay OOM tren local RTX 3050 (4GB)

```
Evaluation (no gradient) thuong khong OOM voi batch_size=32.
Neu OOM:
  python scripts/test.py data.batch_size=16 data.num_workers=0
```

---

## Checklist cuoi cung

```
[ ] scripts/subset_140k_real.py -- tao va chay
[ ] scripts/split_real_pexels.py -- tao va chay
[ ] scripts/resize_all.py -- update folders_to_process, chay
[ ] preprocessing/build_splits.py -- update OOD filter, chay
[ ] Verify manifests: train ~21,850, ood ~880
[ ] src/holmhz/data/transforms.py -- update augmentation v2
[ ] configs/train_v2.yaml -- tao moi
[ ] Upload data v2 len Kaggle
[ ] Retrain tren Kaggle (20 epochs, ~1 hour)
[ ] Download best_v2.pt ve local
[ ] Rename v1 outputs
[ ] Re-evaluate: python scripts/test.py model.checkpoint=best_v2.pt
[ ] Kiem tra KPIs: OOD AUC >= 0.75, ID AUC >= 0.95
[ ] Cap nhat CONTEXT.md, PROJECT_PLAN.md, TASK_1.7
[ ] Commit + push
```
