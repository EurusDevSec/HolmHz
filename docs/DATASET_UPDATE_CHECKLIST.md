# HolmHz — Dataset Update Checklist

> **Mục đích**: Quy trình chuẩn mỗi khi thay đổi dataset (thêm ảnh, bớt ảnh, thay đổi nguồn).
> Đảm bảo KHÔNG phá vỡ pipeline đã xây từ Task 1.1→1.6.
>
> **Nguyên tắc vàng**: Code không đổi (trừ `resize_all.py` và `build_splits.py` nếu thêm folder mới).
> Config không đổi (tạo file MỚI như `train_v3.yaml`). Checkpoint không bị ghi đè (đặt tên khác).

---

## Tổng quan Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUY TRÌNH CẬP NHẬT DATASET                          │
│                                                                         │
│  Bước 1: Chuẩn bị data/raw/           (thêm/xóa ảnh thô)             │
│      ↓                                                                  │
│  Bước 2: resize_all.py                (resize → data/processed/)       │
│      ↓                                                                  │
│  Bước 3: build_splits.py              (tạo manifests JSON mới)         │
│      ↓                                                                  │
│  Bước 4: validate_dataset.py          (kiểm tra không corrupt)         │
│      ↓                                                                  │
│  Bước 5: Tạo train_vX.yaml MỚI       (config cho lần train này)       │
│      ↓                                                                  │
│  Bước 6: pytest                       (đảm bảo không phá code)         │
│      ↓                                                                  │
│  Bước 7: Nén + Upload Kaggle          (zip data/processed/ mới)        │
│      ↓                                                                  │
│  Bước 8: Kaggle notebook (4 cells)    (clone → symlink → W&B → train)  │
│      ↓                                                                  │
│  Bước 9: Đánh giá (test.py)           (ID + OOD metrics)              │
│      ↓                                                                  │
│  Bước 10: Ghi chép kết quả           (CONTEXT.md / CHANGELOG.md)      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Trước khi bắt đầu — Checklist kiểm tra

- [ ] Branch Git mới (ví dụ: `fix/s1/data-update-v3`)
- [ ] Backup `data/manifests/*.json` hiện tại (đã có trong git)
- [ ] Ghi lại lý do thay đổi dataset (vào mục Log bên dưới)
- [ ] Checkpoint cũ (`best.pt`, `best_v2.pt`) vẫn còn — KHÔNG xóa

---

## Bước 1: Chuẩn bị data thô (`data/raw/`)

### 1a. Nếu THÊM ảnh vào nguồn đã có

Chỉ cần copy ảnh mới vào đúng folder:

```
data/raw/real/ffhq/          ← thêm ảnh real
data/raw/fake_diffusion/sd15/ ← thêm ảnh SD
data/raw/ood_test/flux/       ← thêm ảnh Flux OOD
```

**Không cần sửa code.** Sang Bước 2.

### 1b. Nếu THÊM nguồn dữ liệu hoàn toàn mới

1. Tạo folder mới theo quy ước:

```
Training real:       data/raw/real/{tên_nguồn}/
Training fake GAN:   data/raw/fake_gan/{tên_nguồn}/
Training fake Diff:  data/raw/fake_diffusion/{tên_nguồn}/
OOD test:            data/raw/ood_test/{tên_nguồn}/
```

2. **Sửa `scripts/resize_all.py`** — thêm dòng mới vào `folders_to_process`:

```python
# Thêm dòng mới vào danh sách, ĐỪNG xóa dòng cũ
folders_to_process = [
    # ... dòng cũ giữ nguyên ...
    ("real/{tên_nguồn}",           "train/real/{tên_nguồn}"),        # ← THÊM
    ("fake_diffusion/{tên_nguồn}", "train/fake_diffusion/{tên_nguồn}"), # ← THÊM
]
```

3. **Nếu folder OOD mới**: Sửa thêm `build_splits.py` → `scan_ood_folder()` → thêm vào `OOD_LABELS`:

```python
OOD_LABELS = {
    # ... giữ nguyên cũ ...
    "tên_nguồn_mới": 1,  # 1 = fake, 0 = real  ← THÊM
}
```

### 1c. Nếu XÓA ảnh

- Xóa trong `data/raw/` trước
- Xóa folder tương ứng trong `data/processed/` (hoặc để Bước 2 chạy lại resize_all.py)
- Bước 3 (build_splits) sẽ tự tạo manifests mới dựa trên những gì còn trong `data/processed/`

---

## Bước 2: Resize ảnh

```bash
python scripts/resize_all.py
```

**Kiểm tra output:**

- Số ảnh mới được resize (dòng `✅ X resized, Y skipped`)
- Ảnh đã xử lý trước đó sẽ bị skip (nhanh)
- Tổng ảnh khớp kỳ vọng

**⚠️ Script này có tính resume** — chỉ resize ảnh chưa có trong `data/processed/`.
Nếu muốn resize lại toàn bộ, xóa folder đích trước.

---

## Bước 3: Rebuild manifests

```bash
python preprocessing/build_splits.py
```

**Kiểm tra output:**

- Dòng `SPLIT RESULTS:` có số liệu hợp lý
- Train/Val/Test ID chia đúng tỷ lệ 70/15/15
- OOD tách riêng hoàn toàn
- Manifests ghi vào `data/manifests/`

**Sau khi chạy, verify nhanh:**

```bash
python -c "
import json
for name in ['train', 'val', 'test_id', 'test_ood']:
    with open(f'data/manifests/{name}.json') as f:
        data = json.load(f)
    real = sum(1 for e in data if e['label'] == 0)
    fake = sum(1 for e in data if e['label'] == 1)
    sources = sorted(set(e['source'] for e in data))
    print(f'{name:10s}: {len(data):6d} (real={real}, fake={fake}) sources={sources}')
"
```

**Kết quả mẫu (sau Task 1.7):**

```
train     :  20860 (real=10737, fake=10123) sources=['cifake', 'diverse_real', 'ffhq', 'real_pexels_train', 'sd15', 'stylegan']
val       :   4470 (real=2271, fake=2199)   sources=['cifake', 'diverse_real', 'ffhq', 'real_pexels_train', 'sd15', 'stylegan']
test_id   :   4470 (real=2292, fake=2178)   sources=['cifake', 'diverse_real', 'ffhq', 'real_pexels_train', 'sd15', 'stylegan']
test_ood  :    880 (real=300, fake=580)      sources=['flux', 'real_camera', 'real_pexels', 'tristanzhang_fake']
```

---

## Bước 4: Validate data

```bash
python scripts/validate_dataset.py
```

**Expect:** `ALL DATA VALID` — không có ảnh corrupt, wrong size, hoặc 0 bytes.

---

## Bước 5: Tạo config training MỚI

**⚠️ QUAN TRỌNG: KHÔNG sửa config cũ. Tạo file mới.**

```
configs/train.yaml      ← Task 1.6 (v1), KHÔNG CHẠM
configs/train_v2.yaml   ← Task 1.7 (v2), KHÔNG CHẠM
configs/train_v3.yaml   ← Lần này, TẠO MỚI
```

Copy từ version trước và chỉnh sửa:

```bash
cp configs/train_v2.yaml configs/train_v3.yaml
```

Các trường thường cần chỉnh:

- `freeze_backbone` — `true` (chỉ head) hay `false` (toàn bộ)
- `learning_rate` — tốt nhất từ HP tuning trước: `0.0001`
- `epochs` — 20 là đủ cho dataset ~20-30K
- `early_stopping.patience` — 5-7
- `batch_size` — 32 (Kaggle T4), 8 (local RTX 3050)

**Nếu không chắc hyperparameters**: giữ nguyên settings của version thắng trước đó.

---

## Bước 6: Chạy tests

```bash
python -m pytest tests/ -v --tb=short
```

**Expect:** Tất cả tests pass (hiện tại 83 tests). Nếu fail → fix trước khi tiếp.

---

## Bước 7: Nén và Upload lên Kaggle

### 7a. Dọn rác trong `data/processed/`

Kiểm tra chỉ có 2 folder cần thiết:

```powershell
# PowerShell — liệt kê
ls data\processed\

# Chỉ nên thấy:
#   train/      ← tất cả ảnh training
#   ood_test/   ← ảnh OOD test
```

**Xóa folder rác** nếu có (ví dụ: `test_ood/`, `val/`, hay bất kỳ folder lạ):

```powershell
# PowerShell — xóa nếu cần
Remove-Item -Recurse -Force data\processed\test_ood   # folder duplicate
Remove-Item -Recurse -Force data\processed\val         # folder rỗng
```

### 7b. Nén

```powershell
cd data\processed
Compress-Archive -Path train, ood_test -DestinationPath ..\holmhz-processed-vX.zip
cd ..\..
```

> Thay `vX` bằng version hiện tại (v2, v3, ...).

### 7c. Upload Kaggle

1. Vào https://www.kaggle.com/datasets → "New Dataset"
2. Upload file zip → đặt tên `holmhz-processed-vX`
3. Set **Private** nếu chưa muốn công khai

---

## Bước 8: Kaggle Notebook (4 Cells)

Tạo notebook mới trên Kaggle hoặc fork notebook cũ.

### Cell 1 — Clone code

```python
import os
!git clone https://github.com/EurusDevSec/HolmHz.git
os.chdir("HolmHz")
!git checkout {TÊN_BRANCH}   # ← thay bằng branch hiện tại
!pip install hatchling grad-cam --quiet
!pip install . --quiet
!python -c "import holmhz; print('✅ HolmHz installed')"
!ls data/manifests/
```

> **Thay đổi duy nhất**: tên branch tại `git checkout`.

### Cell 2 — Symlink data (KHÔNG CẦN ĐỔI)

```python
import os, json

KAGGLE_INPUT = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train" in dirs:
        train_path = os.path.join(root, "train")
        if os.path.isdir(train_path) and len(os.listdir(train_path)) > 0:
            KAGGLE_INPUT = root
            break

if KAGGLE_INPUT is None:
    raise FileNotFoundError("❌ Không tìm thấy train/ trong /kaggle/input/")

print(f"✅ Found: {KAGGLE_INPUT}")

!rm -rf data/processed
!ln -s {KAGGLE_INPUT} data/processed

# Verify
print("\n=== Symlink check ===")
!ls data/processed/train/ | head -5
!ls data/manifests/

with open("data/manifests/train.json") as f:
    first = json.load(f)[0]
print(f"\nManifest path: {first['path']}")
print(f"File exists:   {os.path.exists(first['path'])}")
```

> Cell này **KHÔNG BAO GIỜ cần thay đổi** — auto-detect sẽ tìm đúng
> thư mục dù bạn upload bao nhiêu lần, đặt tên gì.

### Cell 3 — W&B (KHÔNG CẦN ĐỔI)

```python
import os
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
    print("✅ W&B API key loaded")
except Exception:
    print("⚠️ Set WANDB_API_KEY manually")

!python -c "import wandb; wandb.login(); print('✅ W&B connected')"
```

> Cell này **KHÔNG BAO GIỜ cần thay đổi**.

### Cell 4 — Training

```python
import os, shutil

CKPT_DIR = "outputs/checkpoints"
RESULT_DIR = "/kaggle/working"

os.makedirs(CKPT_DIR, exist_ok=True)
for f in ["best.pt", "last.pt"]:
    p = os.path.join(CKPT_DIR, f)
    if os.path.exists(p): os.remove(p)

# ── Training ──
!python scripts/train.py \
    --config configs/train_vX.yaml \    # ← thay vX
    data.num_workers=4

# ── Lưu checkpoint ──
shutil.copy2(f"{CKPT_DIR}/best.pt", f"{RESULT_DIR}/best_vX.pt")  # ← thay vX
print("✅ DONE — download best_vX.pt từ Kaggle Output tab")
```

> **Thay đổi 2 chỗ**: config `train_vX.yaml` và output `best_vX.pt`.
>
> **Sau khi chạy**: Save Version → "Save & Run All (Commit)" → đi ngủ.
> Kết quả xem trên W&B hoặc Kaggle Output tab sáng hôm sau.

---

## Bước 9: Đánh giá (sau khi có checkpoint)

### 9a. Tải checkpoint về local

1. Kaggle → Notebook → tab "Output"
2. Download `best_vX.pt`
3. Copy vào `outputs/checkpoints/best_vX.pt`

### 9b. Chạy evaluation

```bash
python scripts/test.py \
    model.checkpoint=outputs/checkpoints/best_vX.pt \
    data.num_workers=0 \
    data.batch_size=32
```

### 9c. Kiểm tra kết quả

Mở `outputs/evaluation/eval_report.json`:

```
Metric quan trọng:
  - ID AUC ≥ 0.95      (dữ liệu cùng distribution)
  - OOD AUC ≥ 0.75     (dữ liệu khác distribution)
  - ID Accuracy ≥ 0.90
```

---

## Bước 10: Ghi chép kết quả

Cập nhật `docs/CONTEXT.md` hoặc `docs/CHANGELOG.md` với:

- Version dataset (v1, v2, v3...)
- Số ảnh mỗi loại
- Config dùng (`train_vX.yaml`)
- Val AUC, OOD AUC, Accuracy
- Checkpoint name (`best_vX.pt`)
- Link W&B run

---

## Cấu trúc file naming — Quy ước rõ ràng

| Version | Trigger             | Config          | Checkpoint   | Kaggle Dataset        | Branch                      |
| ------- | ------------------- | --------------- | ------------ | --------------------- | --------------------------- |
| v1      | Task 1.6 (baseline) | `train.yaml`    | `best.pt`    | `holmhz-processed`    | `feat/s1/baseline-training` |
| v2      | Task 1.7 (OOD fix)  | `train_v2.yaml` | `best_v2.pt` | `holmhz-processed-v2` | `fix/s1/ood-improvement`    |
| v3      | (tương lai)         | `train_v3.yaml` | `best_v3.pt` | `holmhz-processed-v3` | `fix/s1/data-update-v3`     |

> **Quy tắc**: Mỗi lần thay đổi dataset → tăng version → tạo file mới.
> KHÔNG BAO GIỜ ghi đè file cũ.

---

## Cấu trúc thư mục — Những gì phải tồn tại

```
data/
├── raw/                       ← ẢNH GỐC (chưa resize, không upload Kaggle)
│   ├── real/
│   │   ├── cifake_subset/     (7,000)
│   │   ├── ffhq/              (5,000)
│   │   ├── diverse_real/      (3,000 — thêm từ v2)
│   │   └── real_pexels_train/ (300 — thêm từ v2)
│   ├── fake_gan/
│   │   └── stylegan/          (5,000)
│   ├── fake_diffusion/
│   │   ├── cifake_subset/     (7,000)
│   │   └── sd15/              (2,500)
│   └── ood_test/
│       ├── tristanzhang_fake/ (500)
│       ├── flux/              (80)
│       ├── real_pexels/       (500)
│       └── real_camera/       (100)
│
├── processed/                 ← ẢNH ĐÃ RESIZE 224×224 (upload Kaggle)
│   ├── train/                 ← CHỈ CÓ 2 FOLDER NÀY
│   │   ├── real/
│   │   ├── fake_gan/
│   │   └── fake_diffusion/
│   └── ood_test/              ← KHÔNG PHẢI test_ood, KHÔNG PHẢI val
│       ├── tristanzhang_fake/
│       ├── flux/
│       ├── real_pexels/
│       └── real_camera/
│
├── manifests/                 ← JSON MANIFESTS (commit vào git)
│   ├── train.json
│   ├── val.json
│   ├── test_id.json
│   ├── test_ood.json
│   ├── dataset_stats.json
│   └── real_pexels_test_only.txt  (filter file — thêm từ v2)
│
└── ❌ KHÔNG NÊN CÓ:
    ├── processed/val/          ← RỖng, vô nghĩa (val dùng ảnh trong train/)
    └── processed/test_ood/     ← Duplicate lỗi (đúng tên là ood_test/)
```

---

## Giải thích: val/ rỗng là ĐÚNG

`val.json` tham chiếu ảnh nằm trong `data/processed/train/`. Đây là thiết kế từ Task 1.3:

- `build_splits.py` chia ảnh theo **manifest** (JSON), không phải theo **folder**
- Ảnh vật lý **tất cả** nằm trong `train/` (vì chúng được resize vào đó)
- `val.json` chỉ là danh sách path trỏ đến 15% ảnh trong `train/` mà không dùng để train

→ **Không cần folder `val/` riêng.**

---

## Giải thích: Tại sao chỉ 1 run thay vì 4?

Task 1.6 chạy 4 runs vì **khám phá** (chưa biết settings nào tốt):

```
Phase 1 (freeze, LR=1e-3)  → Warm-up head             → AUC 0.9419
Phase 2 (unfreeze, LR=1e-4) → Fine-tune full           → AUC 0.9983 ← THẮNG
HP Run A (unfreeze, LR=5e-4) → Test LR cao hơn         → AUC 0.9982
HP Run B (unfreeze, LR=5e-5) → Test LR thấp hơn        → AUC 0.9978
```

Từ Task 1.7 trở đi, đã **biết** `unfreeze + LR=1e-4` là tốt nhất.
→ Chỉ cần 1 run. Nếu kết quả tệ → quay lại HP tuning.

**Khi nào cần chạy lại 4 runs?**

- Thay đổi model architecture (không còn EfficientNet-B0)
- Dataset thay đổi DRASTICALLY (>50% khác biệt)
- Thêm loss function mới hoặc thay đổi augmentation cực mạnh

---

## Troubleshooting — Lỗi thường gặp

### 1. `FileNotFoundError` khi train

```
Nguyên nhân: Manifest JSON trỏ đến ảnh không tồn tại
Fix:         Chạy lại Bước 2 (resize) → Bước 3 (build_splits)
Verify:      python -c "
import json, os
broken = 0
for name in ['train', 'val']:
    with open(f'data/manifests/{name}.json') as f:
        for e in json.load(f):
            if not os.path.exists(e['path']):
                broken += 1
                if broken <= 5: print(f'MISSING: {e[\"path\"]}')
print(f'Total broken paths: {broken}')
"
```

### 2. `last.pt` gây auto-resume nhầm

```
Nguyên nhân: train.py tự động resume từ outputs/checkpoints/last.pt
Fix:         Xóa last.pt TRƯỚC KHI train lần mới
             rm outputs/checkpoints/last.pt
```

### 3. resize_all.py tạo folder `test_ood/` thay vì `ood_test/`

```
Nguyên nhân: resize_all.py output là test_ood/, nhưng build_splits.py
             và manifests đọc từ ood_test/ (folder gốc từ Task 1.2)
Fix:         Xóa test_ood/ (duplicate). ood_test/ mới là folder đúng.
             rm -rf data/processed/test_ood
Lưu ý:      Chỉ là naming mismatch, không ảnh hưởng nếu ood_test/ đã có sẵn.
```

### 4. OOD AUC thấp sau khi thay đổi dataset

```
Nguyên nhân: Data mới có thể không khớp distribution OOD test
Phân tích:
  1. Chạy test.py → xem per-source accuracy
  2. Nếu flux/tristanzhang thấp → cần thêm diffusion data vào training
  3. Nếu real_pexels/real_camera thấp → cần thêm real đa dạng
```

### 5. Tests fail sau khi sửa code

```
Fix: Đọc kỹ error message
  - test_data.py fail → kiểm tra transforms.py hoặc image_dataset.py
  - test_training.py fail → kiểm tra train.yaml format
  - test_detectors.py fail → kiểm tra model config

Luôn chạy: python -m pytest tests/ -v --tb=short
TRƯỚC KHI push lên Kaggle.
```

---

## Dataset Change Log

> Ghi lại mỗi lần thay đổi dataset tại đây.

### v1 — Task 1.6 (Baseline) — 2026-02-26

| Loại              | Số ảnh       | Sources                                                             |
| ----------------- | ------------ | ------------------------------------------------------------------- |
| Train total       | 26,500       | cifake(7K) + ffhq(5K) + stylegan(5K) + cifake_diff(7K) + sd15(2.5K) |
| Manifests train   | 18,550       | 70% of 26,500                                                       |
| Manifests val     | 3,975        | 15%                                                                 |
| Manifests test_id | 3,975        | 15%                                                                 |
| OOD test          | 1,180        | tristanzhang(500) + real_pexels(500) + flux(80) + real_camera(100)  |
| **ID AUC**        | **0.9983**   |                                                                     |
| **OOD AUC**       | **0.4812**   | ← Thất bại mục tiêu ≥0.75                                           |
| Config            | `train.yaml` | Phase2: unfreeze, LR=1e-4                                           |
| Checkpoint        | `best.pt`    |                                                                     |

### v2 — Task 1.7 (OOD Improvement) — 2026-03-01

| Loại              | Số ảnh                                                         | Sources                                                                |
| ----------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Train total       | 29,800                                                         | v1 + diverse_real(3K) + real_pexels_train(300)                         |
| Manifests train   | 20,860                                                         | 70% of 29,800                                                          |
| Manifests val     | 4,470                                                          | 15%                                                                    |
| Manifests test_id | 4,470                                                          | 15%                                                                    |
| OOD test          | 880                                                            | tristanzhang(500) + real_pexels(**200**) + flux(80) + real_camera(100) |
| **ID AUC**        | _chưa chạy_                                                    |                                                                        |
| **OOD AUC**       | _chưa chạy_                                                    | Target ≥ 0.75                                                          |
| Config            | `train_v2.yaml`                                                | unfreeze, LR=1e-4, aug v2                                              |
| Checkpoint        | `best_v2.pt`                                                   |                                                                        |
| Thay đổi          | +3,300 real train, -300 OOD real_pexels, augmentation mạnh hơn |

### v3 — (Template cho lần sau)

| Loại        | Số ảnh          | Sources |
| ----------- | --------------- | ------- |
| Train total |                 |         |
| OOD test    |                 |         |
| **ID AUC**  |                 |         |
| **OOD AUC** |                 | Target: |
| Config      | `train_v3.yaml` |         |
| Checkpoint  | `best_v3.pt`    |         |
| Thay đổi    |                 |         |
