# OOD v5 Optimization Guide — Cải thiện OOD AUC từ 0.78 lên >0.85

> Tạo: 2026-03-03 | Branch: `fix/s1/ood-improvement`
> File này ghi lại chi tiết phân tích + kế hoạch cải thiện OOD performance sau v4.
> Đọc file này để hiểu toàn bộ context mà không cần đọc lại chat history.

---

## 1. Tình trạng hiện tại — v4 Results

### 1.1 Overall Metrics

| Set       | AUC        | Acc    | F1     | Prec   | Recall | N    |
| --------- | ---------- | ------ | ------ | ------ | ------ | ---- |
| In-Domain | **0.9972** | 0.9742 | 0.9740 | 0.9628 | 0.9855 | 4500 |
| OOD       | **0.7838** | 0.7118 | 0.7531 | 0.7222 | 0.7868 | 680  |

### 1.2 OOD Per-Source Breakdown

| Source            | Label | Acc       | N   | Phân tích                                   |
| ----------------- | ----- | --------- | --- | ------------------------------------------- |
| flux              | Fake  | 77.5%     | 80  | Tốt — unseen generator, detect khá          |
| tristanzhang_fake | Fake  | 79.0%     | 300 | Tốt — MJ/DALLE/SD mixed, 200 ảnh train giúp |
| real_pexels       | Real  | 74.5%     | 200 | OK — 25.5% bị FP (real→fake)                |
| **real_camera**   | Real  | **36.0%** | 100 | **BOTTLENECK** — 64% real bị gọi là fake    |

### 1.3 ID Per-Source (vấn đề nhỏ)

| Source             | Acc   | N    | Ghi chú                  |
| ------------------ | ----- | ---- | ------------------------ |
| cifake             | 96.6% | 2100 | OK                       |
| diverse_real       | 97.3% | 450  | OK                       |
| ffhq               | 99.7% | 750  | Excellent                |
| real_pexels_train  | 73.3% | 45   | ⚠️ Low (nhưng N rất nhỏ) |
| sd15               | 98.9% | 375  | OK                       |
| stylegan           | 98.5% | 750  | OK                       |
| tristanzhang_train | 90.0% | 30   | OK (N nhỏ)               |

### 1.4 Training Info v4

- **Checkpoint**: `outputs/checkpoints/best_v4.pt` (epoch 28, 48.5MB)
- **Config**: `configs/train_v4.yaml`
- **Platform**: Kaggle T4 GPU, 30 epochs (~33 phút)
- **Key changes**: WeightedRandomSampler + pos_weight=1.2 + tristanzhang_train 200 ảnh
- **Val AUC**: 0.9969 (best at epoch 28)
- **Guide**: `docs/KAGGLE_TRAINING_V4.md` (7 cells, tested working)

---

## 2. Diagnostic — Tại sao real_camera chỉ 36%?

### 2.1 Root Cause Analysis

**Vấn đề gốc**: Model bias predict FAKE cho unknown images.

```
real_camera: 100 ảnh real → model predict 64 = FAKE, 36 = REAL
→ False Positive Rate = 64%
```

**Nguyên nhân 1 — Training data thiếu camera real-world diversity**:

| Source (Train)    | N     | Loại                             |
| ----------------- | ----- | -------------------------------- |
| ffhq              | 3,500 | Face portraits (70x70 aligned)   |
| diverse_real      | 2,100 | Real nhưng thiên face/portrait   |
| cifake real       | 4,900 | CIFAR-10 (32x32 upscale, object) |
| real_pexels_train | 210   | Real outdoor/landscape → RẤT ÍT  |

**→ Model chưa thấy đủ real outdoor/landscape/object images ở high-res.**

**Nguyên nhân 2 — `real_camera` images là Unsplash portrait/headshot (~400x446)**:

- Resolution nhỏ hơn training images (224x224 resize → ít detail loss)
- Nhưng style khác biệt: non-studio lighting, mobile camera quality
- Model chưa từng thấy "non-face" real images ở distribution này

**Nguyên nhân 3 — pos_weight=1.2 penalize miss FAKE nhiều hơn miss REAL**:

- pos_weight > 1.0 → model thiên predict FAKE khi không chắc
- Giúp fake detection (flux 77.5%, tristanzhang 79%) nhưng hurt real recognition

### 2.2 So sánh OOD AUC qua các version

| Version | OOD AUC    | Key Change                         | real_camera | real_pexels | tristanzhang | flux      |
| ------- | ---------- | ---------------------------------- | ----------- | ----------- | ------------ | --------- |
| v1      | 0.4812     | Baseline                           | 12.0%       | 8.6%        | 87.2%        | 95.0%     |
| v2      | 0.5215     | +diverse_real +real_pexels_train   | 49.0%       | 90.0%       | 17.4%        | 40.0%     |
| v3      | 0.4916     | (zip bug, no tristanzhang)         | 42.0%       | 85.5%       | 23.7%        | 55.0%     |
| **v4**  | **0.7838** | +sampler +pos_weight +tristanzhang | **36.0%**   | 74.5%       | **79.0%**    | **77.5%** |

**Nhận xét**:

- v2 fix real recognition (real_pexels 8%→90%) nhưng fake collapse
- v4 balance cả hai: fake detection tốt (77-79%) + real OK (74.5%)
- `real_camera` luôn là nguồn yếu nhất (12%→49%→42%→36%)
- real_camera giảm từ v2→v4 vì pos_weight + sampler push model predict FAKE nhiều hơn

---

## 3. Kế hoạch cải thiện — 5 Strategies (ưu tiên giảm dần)

### Strategy 1 ⭐ — Thêm Real Camera/Outdoor Data vào Training (IMPACT: HIGH)

**Lý do**: Thiếu training data real outdoor = root cause #1.

**Nguồn data**:

- **Unsplash API** (đang dùng cho real_camera OOD): Download thêm 500-1000 ảnh landscape/object/street
- **Pexels API** (đã có script `split_real_pexels.py`): Có thể download thêm
- **COCO 2017 val set**: 5,000 ảnh real-world camera, đa dạng scene

**Implementation**:

```
data/processed/train/real/
├── cifake/          # 4,900 (existing, label=0)
├── diverse_real/    # 2,100 (existing, label=0)
├── ffhq/            # 3,500 (existing, label=0)
├── real_pexels_train/  # 210 (existing, label=0)
└── real_camera_train/  # 300-500 NEW (label=0)  ← THÊM MỚI
```

**Steps**:

1. Download 500 ảnh từ Unsplash (landscape, street, nature, food, architecture — KHÔNG portrait)
2. Copy 300 vào `data/processed/train/real/real_camera_train/`
3. Giữ 200 cho OOD test (hoặc dùng tập real_camera hiện tại)
4. Resize: `python scripts/resize_all.py` (cần thêm mapping)
5. Rebuild: `python preprocessing/build_splits.py`

**Expected impact**: real_camera 36% → 60-70%, OOD AUC 0.78 → 0.82-0.85

### Strategy 2 — Optimal Threshold (IMPACT: MEDIUM, ZERO COST)

**Lý do**: Threshold=0.5 cố định. Có thể tìm threshold tối ưu trên val set.

**Implementation**:
Thêm vào `scripts/test.py` hoặc tạo `analysis/find_threshold.py`:

```python
from sklearn.metrics import roc_curve
import numpy as np

# Sau khi chạy inference trên val set
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
# Youden's J statistic
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.4f}")
```

Hoặc dùng threshold maximize F1 trên val:

```python
from sklearn.metrics import f1_score
best_f1, best_t = 0, 0.5
for t in np.arange(0.3, 0.7, 0.01):
    preds = (y_scores >= t).astype(int)
    f1 = f1_score(y_true, preds)
    if f1 > best_f1:
        best_f1, best_t = f1, t
```

Sau đó cập nhật `configs/test.yaml`:

```yaml
evaluation:
  threshold: 0.42 # (ví dụ — cần chạy script để tìm)
```

**Expected impact**: OOD AUC không đổi (AUC invariant to threshold), nhưng Accuracy có thể tăng 5-10%. Đặc biệt giúp real_camera nếu model predict probability gần 0.5 cho real images.

### Strategy 3 — Giảm pos_weight hoặc thử pos_weight < 1.0 (IMPACT: MEDIUM)

**Lý do**: pos_weight=1.2 tăng recall for FAKE nhưng giảm precision (hại real recognition).

**Thử nghiệm**:
| Experiment | pos_weight | Expected Effect |
| ------------- | ---------- | --------------------------------------- |
| v4 (current) | 1.2 | Baseline |
| v5a | 1.0 | Bỏ pos_weight → less FAKE bias |
| v5b | 0.8 | Giảm → more REAL bias → help real_camera |

**Implementation**: Chỉ cần sửa `configs/train_v5.yaml`:

```yaml
training:
  pos_weight: 0.8 # hoặc 1.0
```

**Expected impact**: real_camera 36% → 50-60%, nhưng tristanzhang_fake/flux có thể giảm 5-10%.

### Strategy 4 — Label Smoothing (IMPACT: MEDIUM)

**Lý do**: Model quá tự tin (overconfident) → predict extreme probabilities. Label smoothing giúp calibrate.

**Implementation** — Sửa `src/holmhz/training/losses.py`:

```python
def get_loss_fn(name, pos_weight=None, label_smoothing=0.0):
    if name == "bce_with_logits":
        # BCEWithLogitsLoss không hỗ trợ label_smoothing trực tiếp
        # → Smooth labels: y_smooth = y * (1 - alpha) + 0.5 * alpha
        # Tức: 1 → 0.95, 0 → 0.05 (với alpha=0.1)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]) if pos_weight else None
        )
        return loss_fn, label_smoothing  # Apply smoothing in training loop
```

Trong training loop (`src/holmhz/training/trainer.py`):

```python
if label_smoothing > 0:
    labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
```

**Config**:

```yaml
training:
  label_smoothing: 0.1 # 1→0.95, 0→0.05
```

**Expected impact**: Model ít overconfident → threshold optimization hiệu quả hơn. OOD AUC +0.02-0.05.

### Strategy 5 — Test-Time Augmentation (TTA) (IMPACT: LOW-MEDIUM, ZERO TRAINING COST)

**Lý do**: Ensemble nhiều augmented views → predictions ổn định hơn.

**Implementation** — Dùng library `ttach` (đã cài):

```python
import ttach as tta

# Wrap model
tta_model = tta.ClassificationTTAWrapper(
    model,
    transforms=tta.Compose([
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.9, 1.0, 1.1]),
    ]),
    merge_mode='mean'
)

# Inference
with torch.no_grad():
    output = tta_model(images)
```

Hoặc manual TTA:

```python
def predict_with_tta(model, image_tensor, n_augments=5):
    """Average predictions over augmented copies."""
    preds = []
    preds.append(model(image_tensor).sigmoid())
    preds.append(model(torch.flip(image_tensor, [3])).sigmoid())  # H-flip
    # ... more augments
    return torch.stack(preds).mean(0)
```

**Expected impact**: OOD AUC +0.01-0.03. Nhưng inference chậm hơn x3-5.

---

## 4. Recommended v5 Plan

### Phase A — Quick Wins (không cần retrain)

1. **[Strategy 2]** Tìm optimal threshold trên val set → cập nhật `configs/test.yaml`
2. **[Strategy 5]** Thử TTA trên OOD test → xem improvement bao nhiêu
3. Nếu đủ target (>0.85) → DONE

### Phase B — Retrain (nếu Phase A chưa đủ)

4. **[Strategy 1]** Download 500 real camera/outdoor images → thêm vào training
5. **[Strategy 3]** Giảm pos_weight: 1.2 → 1.0 (hoặc 0.8)
6. **[Strategy 4]** Thêm label_smoothing: 0.1
7. Train v5 trên Kaggle (dùng `KAGGLE_TRAINING_V4.md` làm template, sửa config)

### Estimated Timeline

| Phase     | Task                         | Thời gian   |
| --------- | ---------------------------- | ----------- |
| A1        | Optimal threshold            | 30 phút     |
| A2        | TTA evaluation               | 30 phút     |
| B1        | Download + preprocess images | 1-2 giờ     |
| B2        | Tạo holmhz-data-v4.zip       | 30 phút     |
| B3        | Train v5 trên Kaggle         | 45 phút     |
| B4        | Evaluate + document          | 30 phút     |
| **Total** | **Phase A + B**              | **3-5 giờ** |

---

## 5. Files cần sửa / tạo

### Nếu chỉ Phase A (no retrain):

| File                         | Action | Mô tả                        |
| ---------------------------- | ------ | ---------------------------- |
| `analysis/find_threshold.py` | NEW    | Script tìm optimal threshold |
| `scripts/test.py`            | EDIT   | Thêm optional TTA support    |
| `configs/test.yaml`          | EDIT   | threshold: 0.5 → optimal     |

### Nếu Phase B (retrain):

| File                              | Action | Mô tả                               |
| --------------------------------- | ------ | ----------------------------------- |
| `scripts/download_real_camera.py` | NEW    | Download Unsplash outdoor images    |
| `scripts/resize_all.py`           | EDIT   | +real_camera_train mapping          |
| `preprocessing/build_splits.py`   | EDIT   | +real_camera filter (nếu cần)       |
| `configs/train_v5.yaml`           | NEW    | pos_weight=1.0, label_smoothing=0.1 |
| `src/holmhz/training/losses.py`   | EDIT   | +label_smoothing support            |
| `src/holmhz/training/trainer.py`  | EDIT   | Apply label smoothing to targets    |
| `docs/KAGGLE_TRAINING_V5.md`      | NEW    | Updated Kaggle guide                |
| `_create_kaggle_zip.py`           | EDIT   | Include new data                    |

---

## 6. Training Data Distribution (hiện tại)

### Train (21,000 samples)

| Source             | N          | Label | % Total | Ghi chú                         |
| ------------------ | ---------- | ----- | ------- | ------------------------------- |
| cifake (real part) | 4,900      | Real  | 23.3%   | CIFAR-10 32x32 upscale          |
| cifake (fake part) | 4,900      | Fake  | 23.3%   | CIFAR-10 AI generates           |
| ffhq               | 3,500      | Real  | 16.7%   | Face portraits aligned          |
| stylegan           | 3,500      | Fake  | 16.7%   | GAN face                        |
| diverse_real       | 2,100      | Real  | 10.0%   | Mixed real (ImageNet)           |
| sd15               | 1,750      | Fake  | 8.3%    | Stable Diffusion 1.5            |
| real_pexels_train  | 210        | Real  | 1.0%    | Pexels outdoor                  |
| tristanzhang_train | 140        | Fake  | 0.7%    | MJ/DALLE/SD mixed               |
| **Total**          | **21,000** |       |         | **Real: 10,737 / Fake: 10,263** |

### WeightedRandomSampler Effect

Với sampler, effective distribution mỗi epoch:

- cifake: 9800 → ~3000 (giảm)
- ffhq: 3500 → ~3000
- stylegan: 3500 → ~3000
- diverse_real: 2100 → ~3000 (tăng)
- sd15: 1750 → ~3000 (tăng)
- real_pexels_train: 210 → **~3000** (tăng mạnh x14)
- tristanzhang_train: 140 → **~3000** (tăng mạnh x21)

→ Mỗi source ~3000 effective samples / epoch (with replacement).

### OOD Test (680 samples)

| Source            | N   | Label | Generator                      |
| ----------------- | --- | ----- | ------------------------------ |
| tristanzhang_fake | 300 | Fake  | Midjourney + DALL-E + SD mixed |
| real_pexels       | 200 | Real  | Pexels/Unsplash natural photos |
| real_camera       | 100 | Real  | Unsplash portrait/headshot     |
| flux              | 80  | Fake  | FLUX.1-schnell                 |

---

## 7. Kaggle Training Reference

### Hiện tại dùng `docs/KAGGLE_TRAINING_V4.md` — 7 cells:

1. **Cell 1**: pip install deps
2. **Cell 2**: Auto-detect + copy data (rmtree+copytree)
3. **Cell 3**: Tạo tristanzhang_train + rebuild manifests
4. **Cell 4**: `sys.path.insert(0, "src")` (thay vì pip install -e .)
5. **Cell 5**: Ghi `configs/train_v4.yaml` inline
6. **Cell 6**: Train (`PYTHONPATH=src python scripts/train.py`)
7. **Cell 7**: Copy checkpoint + OOD eval

### Cho v5 — cần sửa:

- Cell 3: Thêm logic copy real_camera_train (nếu Strategy 1)
- Cell 5: Ghi `configs/train_v5.yaml` (sửa pos_weight, thêm label_smoothing)
- Phần còn lại giữ nguyên

### Lưu ý kỹ thuật Kaggle:

- **hatchling không hoạt động** trên Kaggle → phải dùng `sys.path.insert(0, "src")`
- **PYTHONPATH=src** phải prefix trước mọi `!python scripts/...` command
- **Auto-detect path**: Dùng `Path("/kaggle/input").rglob("src")` vì path thay đổi
- **Luôn delete + recopy**: `shutil.rmtree(dst)` trước `shutil.copytree(src, dst)`

---

## 8. Checkpoints Available

| File                             | Version    | Epoch | Val AUC | OOD AUC    | Notes                   |
| -------------------------------- | ---------- | ----- | ------- | ---------- | ----------------------- |
| `outputs/checkpoints/best.pt`    | v1         | 12    | 0.9983  | 0.4812     | Baseline                |
| `outputs/checkpoints/best_v2.pt` | v2         | 16    | 0.9970  | 0.5215     | +diverse_real           |
| `outputs/checkpoints/best_v3.pt` | v3 (wrong) | 21    | 0.9974  | 0.4916     | Thiếu tristanzhang      |
| `outputs/checkpoints/best_v4.pt` | **v4** ✅  | 28    | 0.9969  | **0.7838** | Sampler+pos_weight+data |

---

## 9. Key Code Locations

| Component             | File                              | What                                                    |
| --------------------- | --------------------------------- | ------------------------------------------------------- |
| WeightedRandomSampler | `src/holmhz/data/utils.py:18-80`  | `compute_source_weights()`, `create_weighted_sampler()` |
| DataLoader factory    | `src/holmhz/data/utils.py:83-142` | `create_dataloader(use_weighted_sampler=)`              |
| Loss function         | `src/holmhz/training/losses.py`   | `get_loss_fn(pos_weight=)`                              |
| Training loop         | `src/holmhz/training/trainer.py`  | `Trainer.train_epoch()`, `Trainer.validate()`           |
| Train script          | `scripts/train.py`                | CLI entry (positional config arg!)                      |
| Test script           | `scripts/test.py`                 | Eval ID + OOD, saves report + plots                     |
| Augmentations         | `src/holmhz/data/transforms.py`   | Train v2 augments (aggressive)                          |
| Build splits          | `preprocessing/build_splits.py`   | Scan processed/ → manifests JSON                        |
| Config train v4       | `configs/train_v4.yaml`           | sampler=true, pos_weight=1.2, 30 epochs                 |
| Config test           | `configs/test.yaml`               | threshold=0.5, checkpoint path                          |

---

## 10. Lịch sử thay đổi quan trọng

### v4 (03/03/2026) — Task 1.7 Completion

**Code changes**:

- `src/holmhz/data/utils.py`: +`compute_source_weights()`, +`create_weighted_sampler()`, `use_weighted_sampler` param
- `src/holmhz/data/__init__.py`: +exports
- `scripts/train.py`: +sampler support, +pos_weight support via OmegaConf
- `configs/train_v4.yaml`: NEW
- `tests/test_data.py`: +3 tests (83/83 total pass)
- `docs/KAGGLE_TRAINING_V4.md`: NEW (7 cells, 8 rounds of fixes)
- `_create_kaggle_zip.py`: NEW (31,238 files, 2.0 GB zip)

**Kaggle fixes applied**:

1. Cell 2: Auto-detect dataset path via `rglob("src")`
2. Cell 2: Always delete+recopy (not skip-if-exists)
3. Cell 3: Set complement for filter file (not sorted slice)
4. Cell 4: `sys.path.insert(0, "src")` instead of `pip install -e .`
5. Cell 6/7: `PYTHONPATH=src` prefix for subprocess commands

### v1-v3 history — see `docs/CONTEXT.md` Section 17

---

## 11. Quyết định thiết kế & Tradeoffs

### Tại sao chọn WeightedRandomSampler thay vì oversample/undersample?

- **Oversample** (duplicate minority → same size as majority): Overfitting minority
- **Undersample** (drop majority → same size as minority): Mất data
- **WeightedRandomSampler** (replacement=True): Mỗi epoch, mỗi source ~equal effective samples. Minority được lặp lại nhưng khác augmentation mỗi lần → less overfitting.

### Tại sao pos_weight=1.2 mà không phải 2.0?

- pos_weight=1.0: No bias → model tự decide
- pos_weight=1.2: Nhẹ push predict FAKE → giúp recall fake nhưng ít hại real
- pos_weight=2.0: Quá mạnh → model sẽ predict FAKE cho mọi thứ

### Tại sao chọn 200 tristanzhang_train (không phải 400)?

- Tổng tristanzhang_fake: 500 ảnh
- Split: 200 train / 300 test
- Nếu 400 train → chỉ 100 test → OOD test set quá nhỏ, unreliable metrics
- 200 đủ để teach model "high-quality fakes exist" qua sampler (200 → ~3000 effective)
