# ===========================================
# HolmHz Training v4 — Final OOD Improvement
# ===========================================
# Chạy trên Kaggle T4 GPU
#
# Thay đổi chính:
# 1. WeightedRandomSampler: cân bằng 7 sources (downsample cifake 47%→14%)
# 2. pos_weight=1.2: nhẹ nhàng boost fake detection
# 3. tristanzhang_train TRONG training data (đã thiếu 2 lần trước)
# 4. epochs=30, patience=10
#
# CHECKLIST trước khi chạy:
# [ ] Dataset "holmhz-data-v3" đã upload (bao gồm processed/ + manifests/)
# [ ] WANDB_API_KEY đã set
# [ ] GPU T4 đã bật

# %% [markdown]
# ## Cell 1: Setup & Install

# %%
!pip install -q wandb omegaconf timm albumentations tqdm python-dotenv rich scipy scikit-learn

import os
os.environ['WANDB_API_KEY'] = 'YOUR_WANDB_KEY_HERE'  # <-- THAY KEY CỦA BẠN

# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Cell 2: Mount Dataset + Copy Code & Data

# %%
import shutil
from pathlib import Path

# === Kaggle dataset path (update tên dataset của bạn) ===
DATA_INPUT = Path("/kaggle/input/holmhz-data-v3")  # <-- UPDATE NẾU TÊN KHÁC

# --- Copy code files (src, scripts, preprocessing, configs, pyproject.toml) ---
code_dirs = ["src", "scripts", "preprocessing", "configs"]
code_files = ["pyproject.toml"]

for d in code_dirs:
    src = DATA_INPUT / d
    dst = Path(d)
    if not dst.exists() and src.exists():
        print(f"Copying {src} → {dst}...")
        shutil.copytree(src, dst)
    elif dst.exists():
        print(f"  {dst} already exists, skipping")
    else:
        print(f"  WARNING: {src} not found!")

for f in code_files:
    src = DATA_INPUT / f
    dst = Path(f)
    if not dst.exists() and src.exists():
        print(f"Copying {src} → {dst}")
        shutil.copy2(src, dst)
    elif dst.exists():
        print(f"  {dst} already exists, skipping")

# --- Copy data (processed images + manifests) ---
for folder in ["processed", "manifests"]:
    src = DATA_INPUT / "data" / folder
    dst = Path(f"data/{folder}")
    if not dst.exists() and src.exists():
        print(f"Copying {src} → {dst}...")
        shutil.copytree(src, dst)
        print(f"  Done!")
    elif dst.exists():
        print(f"  {dst} already exists, skipping")
    else:
        print(f"  WARNING: {src} not found!")

# Verify basic structure
print("\n=== Data structure ===")
!find data/processed/train -maxdepth 2 -type d | head -20
!echo "---"
!find data/processed -name "*.png" | wc -l
print("total PNG files in processed/")

# %% [markdown]
# ## Cell 3: Create tristanzhang_train + Rebuild Manifests
#
# **CRITICAL**: Bước này tạo tristanzhang_train data inline.
# 2 lần train trước THIẾU bước này → model không thấy high-quality fakes.

# %%
import random
import json
from pathlib import Path

# === Step 1: Tạo tristanzhang_train từ ood_test ===
OOD_DIR = Path("data/processed/ood_test/tristanzhang_fake")  # Có sẵn trong dataset (500 ảnh)
TRAIN_DIR = Path("data/processed/train/fake_diffusion/tristanzhang_train")
SEED = 42

if not TRAIN_DIR.exists() or len(list(TRAIN_DIR.glob("*.png"))) == 0:
    print("=== Creating tristanzhang_train (200 images from OOD) ===")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    all_images = sorted(list(OOD_DIR.glob("*.png")))
    print(f"Source OOD images: {len(all_images)}")

    random.seed(SEED)
    random.shuffle(all_images)

    train_images = all_images[:200]
    test_images = all_images[200:]

    # Copy 200 → train
    for img in train_images:
        shutil.copy2(img, TRAIN_DIR / img.name)

    print(f"Copied {len(train_images)} images to {TRAIN_DIR}")
    print(f"Remaining {len(test_images)} images stay as OOD test")

    # === Step 2: Write filter files ===
    manifests_dir = Path("data/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # tristanzhang_test_only.txt (300 filenames)
    # Convert .png stems back to .jpg names (original format)
    test_names = sorted([f"{img.stem}.jpg" for img in test_images])
    with open(manifests_dir / "tristanzhang_test_only.txt", "w") as f:
        for name in test_names:
            f.write(name + "\n")
    print(f"Written tristanzhang_test_only.txt: {len(test_names)} entries")

    # real_pexels_test_only.txt (200 filenames) — từ dataset hoặc tạo lại
    PEXELS_DIR = Path("data/processed/ood_test/real_pexels")
    PEXELS_TRAIN_DIR = Path("data/processed/train/real/real_pexels_train")

    pexels_filter = manifests_dir / "real_pexels_test_only.txt"
    if not pexels_filter.exists():
        all_pexels = sorted(list(PEXELS_DIR.glob("*.png")))
        random.seed(SEED)
        random.shuffle(all_pexels)
        # 300 train, 200 test
        pexels_test = all_pexels[300:]
        test_pexels_names = sorted([f"{img.stem}.jpg" for img in pexels_test])
        with open(pexels_filter, "w") as f:
            for name in test_pexels_names:
                f.write(name + "\n")
        print(f"Written real_pexels_test_only.txt: {len(test_pexels_names)} entries")
    else:
        print("real_pexels_test_only.txt already exists")
else:
    n = len(list(TRAIN_DIR.glob("*.png")))
    print(f"tristanzhang_train already exists: {n} images")

# Verify
print(f"\n=== Verification ===")
print(f"tristanzhang_train: {len(list(TRAIN_DIR.glob('*.png')))} images")
print(f"tristanzhang_test_only.txt exists: {Path('data/manifests/tristanzhang_test_only.txt').exists()}")
print(f"real_pexels_test_only.txt exists: {Path('data/manifests/real_pexels_test_only.txt').exists()}")

# === Step 3: Rebuild manifests ===
print("\n=== Rebuilding manifests ===")
!python preprocessing/build_splits.py

# === Step 4: CRITICAL VERIFY ===
print("\n" + "=" * 60)
print("CRITICAL VERIFICATION")
print("=" * 60)

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
        print(f'  {s:25s} real={c["real"]:5d}  fake={c["fake"]:5d}  total={total:5d}')

# Check assertions
train_data = json.load(open('data/manifests/train.json'))
train_sources = set(x['source'] for x in train_data)
assert 'tristanzhang_train' in train_sources, "ERROR: tristanzhang_train NOT in training data!"
assert len(train_data) == 21000, f"ERROR: Expected 21000 train samples, got {len(train_data)}"

ood_data = json.load(open('data/manifests/test_ood.json'))
assert len(ood_data) == 680, f"ERROR: Expected 680 OOD samples, got {len(ood_data)}"

print("\n✅ ALL CHECKS PASSED!")
print(f"  Train: {len(train_data)} samples (with tristanzhang_train)")
print(f"  OOD:   {len(ood_data)} samples")
print("\n🚀 Ready to train!")

# %% [markdown]
# ## Cell 4: Install HolmHz Package

# %%
!pip install -e . --no-deps
import holmhz
print(f"HolmHz version: {holmhz.__version__}")

# Verify augmentation v2
from holmhz.data.transforms import get_train_transforms
t = get_train_transforms(224)
print(f"\nTrain transforms:\n{t}")

# Verify new features
from holmhz.data.utils import compute_source_weights
weights = compute_source_weights('data/manifests/train.json')
print(f"\nSample weights computed: {len(weights)} total")

# Show effective distribution with WeightedSampler
from collections import Counter
data = json.load(open('data/manifests/train.json'))
source_counts = Counter(item['source'] for item in data)
max_count = max(source_counts.values())
print(f"\nSource balance (WeightedSampler):")
for src, cnt in sorted(source_counts.items()):
    w = max_count / cnt
    print(f"  {src:25s} count={cnt:5d}  weight={w:6.2f}  → ~14.3% each epoch")

# %% [markdown]
# ## Cell 5: Write train_v4.yaml config

# %%
config_content = """
# HolmHz Training v4 — Final OOD Improvement
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
with open("configs/train_v4.yaml", "w") as f:
    f.write(config_content)

print("✅ configs/train_v4.yaml written")
!cat configs/train_v4.yaml

# %% [markdown]
# ## Cell 6: TRAIN! 🚀
#
# **Dự kiến**: ~30-45 phút trên T4 GPU
# **Monitor**: W&B dashboard cho val_auc
# **Expected**: val_auc ~0.97-0.99, kết quả OOD sẽ đánh giá sau

# %%
# Xóa checkpoint cũ (train fresh, không resume)
import os
for f in ["outputs/checkpoints/last.pt", "outputs/checkpoints/best.pt"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed old: {f}")

print("Starting training v4...")
print("=" * 60)

!python scripts/train.py configs/train_v4.yaml data.num_workers=4

# %% [markdown]
# ## Cell 7: Copy checkpoint + Quick eval

# %%
import shutil
from pathlib import Path

# Copy best.pt → best_v4.pt
best_src = Path("outputs/checkpoints/best.pt")
if best_src.exists():
    shutil.copy2(best_src, "/kaggle/working/best_v4.pt")
    size_mb = best_src.stat().st_size / 1e6
    print(f"✅ Copied best.pt → /kaggle/working/best_v4.pt ({size_mb:.1f} MB)")
else:
    print("⚠️  best.pt not found! Check training output above.")

# Also copy last.pt as backup
last_src = Path("outputs/checkpoints/last.pt")
if last_src.exists():
    shutil.copy2(last_src, "/kaggle/working/last_v4.pt")
    print(f"✅ Copied last.pt → /kaggle/working/last_v4.pt")

# Quick inline evaluation on OOD
print("\n=== Quick OOD Evaluation ===")
!python scripts/test.py model.checkpoint=outputs/checkpoints/best.pt data.num_workers=4 data.batch_size=32

# %% [markdown]
# ## Cell 8: Download Results
#
# Sau khi chạy xong:
# 1. Download `best_v4.pt` từ Kaggle Output
# 2. Copy vào local: `outputs/checkpoints/best_v4.pt`
# 3. Re-evaluate local:
#    ```
#    python scripts/test.py model.checkpoint=outputs/checkpoints/best_v4.pt data.num_workers=0 data.batch_size=32
#    ```

# %%
print("=== Files to download ===")
!ls -la /kaggle/working/*.pt 2>/dev/null || echo "No .pt files found"
print("\n=== Done! Download best_v4.pt and evaluate locally ===")
