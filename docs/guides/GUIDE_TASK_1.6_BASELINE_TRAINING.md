# 📖 HƯỚNG DẪN CHI TIẾT TASK 1.6: BASELINE TRAINING

> **Dành cho**: Lê Văn Hoàng — người chưa có nền tảng ML/DL, học qua thực hành  
> **Triết lý**: Mỗi bước không chỉ hướng dẫn **làm gì** mà giải thích **tại sao làm vậy**  
> **Thời gian**: ~3-4 ngày  
> **Tiền đề**: Task 1.5 Training Pipeline ✅ (16/16 tests, dry run AUC 0.92)  
> **Tham chiếu**: [TASK_1.6_BASELINE_TRAINING.md](../tasks/TASK_1.6_BASELINE_TRAINING.md) | [PROJECT_PLAN.md](../PROJECT_PLAN.md) Section 6
>
> **Output**: Model checkpoint tốt nhất (`best.pt`) + W&B dashboard + smoke test trên ảnh thật

---

## 📋 Mục lục

- [Bức tranh tổng thể: Baseline Training nằm ở đâu?](#bức-tranh-tổng-thể-baseline-training-nằm-ở-đâu)
- [Tại sao cần 2 Phase Training?](#tại-sao-cần-2-phase-training)
- [Kiến thức nền: Transfer Learning 2 Phase](#kiến-thức-nền-transfer-learning-2-phase)
- [Kiến thức nền: Hyperparameter Tuning](#kiến-thức-nền-hyperparameter-tuning)
- [Kiến thức nền: Overfitting vs Underfitting](#kiến-thức-nền-overfitting-vs-underfitting)
- [Kiến thức nền: Đọc W&B Dashboard](#kiến-thức-nền-đọc-wb-dashboard)
- [Tổng quan các bước](#tổng-quan-các-bước)
- [Bước 0: Chuẩn bị Git branch](#bước-0-chuẩn-bị-git-branch)
- [Bước 1: Dọn dẹp dry run cũ](#bước-1-dọn-dẹp-dry-run-cũ)
- [Bước 2: Phase 1 — Freeze backbone, train head](#bước-2-phase-1--freeze-backbone-train-head)
- [Bước 3: Phân tích kết quả Phase 1](#bước-3-phân-tích-kết-quả-phase-1)
- [Bước 4: Phase 2 — Unfreeze, fine-tune toàn bộ](#bước-4-phase-2--unfreeze-fine-tune-toàn-bộ)
- [Bước 5: Hyperparameter tuning (Phase 2)](#bước-5-hyperparameter-tuning-phase-2)
- [Bước 6: Chọn best model + phân tích](#bước-6-chọn-best-model--phân-tích)
- [Bước 7: Implement predict.py (smoke test)](#bước-7-implement-predictpy-smoke-test)
- [Bước 8: Smoke test trên imgs/](#bước-8-smoke-test-trên-imgs)
- [Bước 9: Document results (CONTEXT.md)](#bước-9-document-results-contextmd)
- [Bước 10: Commit & PR](#bước-10-commit--pr)
- [Checklist hoàn thành](#checklist-hoàn-thành)
- [Troubleshooting](#troubleshooting)
- [Mối liên hệ với các Task tiếp theo](#mối-liên-hệ-với-các-task-tiếp-theo)

---

## Bức tranh tổng thể: Baseline Training nằm ở đâu?

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        DỰ ÁN HOLMHZ — SPRINT 1                          │
│                                                                           │
│  Task 1.1  Setup môi trường ✅ DONE                                      │
│  Task 1.2  Thu thập dữ liệu ✅ DONE (27,680 ảnh)                        │
│  Task 1.3  Data Pipeline    ✅ DONE (17/17 tests, 18,550 train)          │
│  Task 1.4  Model Architecture ✅ DONE (30/30 tests, 4M params)           │
│  Task 1.5  Training Pipeline ✅ DONE (16/16 tests, dry run AUC 0.92)     │
│    │                                                                      │
│    │  Tất cả "công cụ" đã sẵn sàng:                                      │
│    │    ✅ DataLoader trả batch [B, 3, 224, 224]                          │
│    │    ✅ Model forward: images → logits [B, 1]                          │
│    │    ✅ Trainer: train loop + val + checkpoint + W&B                   │
│    │    ✅ W&B connected (hoangslevan-thu-dau-mot-university/holmhz)      │
│    │                                                                      │
│  ► Task 1.6  BASELINE TRAINING  ◄◄◄  BẠN ĐANG Ở ĐÂY                    │
│    │                                                                      │
│    │  Đây là lúc THỰC SỰ TRAIN MODEL — "bấm nút chạy thật"             │
│    │  Không cần code mới (phần lớn), chỉ chạy train.py                  │
│    │  với các config khác nhau và phân tích kết quả.                     │
│    │                                                                      │
│    │  5 việc chính:                                                       │
│    │    1. Phase 1: Freeze backbone, train head (10 epochs)              │
│    │    2. Phase 2: Unfreeze, fine-tune toàn bộ (20 epochs)              │
│    │    3. HP tuning: Thử 3 giá trị LR cho Phase 2                      │
│    │    4. Smoke test: Thử model trên 10 ảnh thật                        │
│    │    5. Document: Ghi lại kết quả                                     │
│    │                                                                      │
│    └──► Sprint 2: Evaluation + XAI + Benchmark                           │
│              │                                                            │
│              │  Model tốt nhất → evaluate OOD, so sánh 3 SOTA            │
│              │  Tích hợp Grad-CAM (giải thích model nhìn vùng nào)       │
│              ▼                                                            │
│         Task 2.1, 2.2, 2.3, 2.4                                          │
│                                                                           │
│  ⚡ Milestone 1: Dataset ≥15k ✅ + Baseline AUC ≥ 0.85                  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Tại sao cần 2 Phase Training?

Nhớ lại từ Task 1.4: EfficientNet-B0 có **4,008,829 params** chia làm 2 phần:

```
┌─────────────────── EfficientNet-B0 ──────────────────┐
│                                                       │
│  BACKBONE: 4,007,548 params (99.97%)                  │
│  ────────────────────────────────────                  │
│  • Đã học từ ImageNet (1.2M ảnh, 1000 lớp)            │
│  • Biết nhận diện: cạnh, texture, hình dạng, vật thể │
│  • Kiến thức "tổng quát" về thị giác máy tính        │
│  • KHÔNG biết gì về deepfake                          │
│                                                       │
│  HEAD: 1,281 params (0.03%)                           │
│  ────────────────────────                             │
│  • Linear(1280, 1) + bias                             │
│  • Chưa được train (random weights)                   │
│  • Sẽ học: "features nào = Fake, features nào = Real" │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Phase 1 — Tại sao freeze backbone trước?

```
TƯƠNG TỰ ĐỜI THỰC:

Bạn thuê một NHIẾP ẢNH GIA chuyên nghiệp (backbone) để GIÁM ĐỊNH ảnh.
Nhiếp ảnh gia đã có "mắt thẩm mỹ" (pretrained kỹ năng nhìn ảnh).

Phase 1: Bạn CHỈ DẠY họ quy tắc mới:
  "Nếu thấy texture lạ → Fake, nếu tự nhiên → Real"
  Không thay đổi "mắt thẩm mỹ" cũ → chỉ thêm quy tắc mới ở đầu ra.

Phase 2: Bạn cho họ TẬP LUYỆN thêm để "mắt" nhạy hơn với deepfake:
  "Hãy tinh chỉnh cách bạn NHÌN ảnh — chú ý vào artifact Diffusion"
  → Cả "mắt" và "quy tắc" đều được cập nhật cùng lúc.
```

**Lý do kỹ thuật:**

| Yếu tố              | Phase 1 (Freeze)                  | Phase 2 (Unfreeze)                        |
| ------------------- | --------------------------------- | ----------------------------------------- |
| Trainable params    | 1,281 (head only)                 | 4,008,829 (toàn bộ)                       |
| Learning rate       | 1e-3 (cao)                        | 1e-4 (thấp 10×)                           |
| Tốc độ train        | Rất nhanh (~2 min/epoch)          | Chậm hơn (~8 min/epoch)                   |
| Nguy cơ overfitting | Thấp (ít params)                  | Cao hơn (nhiều params)                    |
| Mục đích            | Head nhanh chóng học Real vs Fake | Backbone tinh chỉnh features cho deepfake |
| Kỳ vọng AUC         | ≥ 0.90                            | ≥ 0.93                                    |

---

## Kiến thức nền: Transfer Learning 2 Phase

### Freeze là gì?

```python
# Freeze = tắt gradient cho backbone
# → optimizer KHÔNG cập nhật backbone weights
# → chỉ cập nhật head weights

model.backbone.freeze()
# Tương đương:
for param in model.backbone.parameters():
    param.requires_grad = False

# Kiểm tra:
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(trainable)  # → 1,281 (chỉ head)
```

### Unfreeze là gì?

```python
# Unfreeze = bật gradient cho toàn bộ model
# → optimizer CẬP NHẬT cả backbone + head
# → model "sâu hơn" — backbone thay đổi cách extract features

model.backbone.unfreeze()
# Tương đương:
for param in model.backbone.parameters():
    param.requires_grad = True

# Kiểm tra:
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(trainable)  # → 4,008,829 (toàn bộ)
```

### Tại sao LR thấp hơn khi unfreeze?

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    LEARNING RATE — HIỂU ĐƠN GIẢN                        │
│                                                                          │
│  LR = "bước chân" khi đi tìm đáy thung lũng (loss thấp nhất)          │
│                                                                          │
│  Phase 1 (LR=1e-3 = 0.001):                                             │
│    • Chỉ di chuyển 1,281 params → bước lớn OK                           │
│    • Head cần học nhanh từ random → hữu ích                             │
│    • Như "chạy nhanh" khi đường thẳng, ít chướng ngại                   │
│                                                                          │
│  Phase 2 (LR=1e-4 = 0.0001):                                            │
│    • Di chuyển 4,008,829 params → bước phải nhỏ                         │
│    • Backbone đã có pretrained weights TỐT → sửa nhẹ thôi              │
│    • Bước quá lớn → PHÁ HỎY pretrained knowledge                       │
│    • Như "đi chậm cẩn thận" khi đã gần đích                            │
│                                                                          │
│  ⚠️ LR quá cao khi unfreeze → catastrophic forgetting                   │
│     (Model quên hết kiến thức ImageNet, AUC TỤT)                        │
│                                                                          │
│  Kí hiệu: 1e-3 = 0.001, 1e-4 = 0.0001, 5e-5 = 0.00005                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: Hyperparameter Tuning

### Hyperparameter là gì?

Hyperparameters (HP) là các giá trị **bạn chọn trước khi train**, model KHÔNG tự học:

```
┌─────────────────────────────────────────────────────────────┐
│  PARAMETERS (Model tự học)    vs    HYPERPARAMETERS (Bạn chọn)   │
│  ──────────────────────────        ────────────────────────────   │
│  • Weights (4M cái)                 • Learning rate (1e-3, 1e-4) │
│  • Biases                           • Batch size (8, 16, 32)     │
│  • Thay đổi mỗi iteration          • Epochs (10, 20, 30)        │
│  • Model cập nhật qua gradient      • Dropout (0.3)              │
│                                     • Weight decay (1e-4)         │
│                                     • Patience (5)                │
│  → Kết quả TRAINING                 → Bạn THỬ & CHỌN            │
└─────────────────────────────────────────────────────────────┘
```

### Tuning strategy — Grid Search đơn giản

```
Grid Search = thử tất cả tổ hợp, chọn cái tốt nhất

Trong Task 1.6, ta chỉ tune LR cho Phase 2 (3 giá trị):

Run 1: LR = 5e-4  (0.0005)  → "bước trung bình"
Run 2: LR = 1e-4  (0.0001)  → "bước nhỏ" (default)
Run 3: LR = 5e-5  (0.00005) → "bước rất nhỏ"

Giữ cố định: batch=32, optimizer=AdamW, scheduler=cosine, dropout=0.3

Tại sao không tune tất cả HP?
→ Mỗi run tốn 1-3h. 3 LR × 3 batch × 3 dropout = 27 runs = 27-81h
→ Không đủ GPU quota! Tune LR là ĐỦ cho baseline.
```

---

## Kiến thức nền: Overfitting vs Underfitting

```
┌──────────────────────────────────────────────────────────────────────────┐
│                OVERFITTING vs UNDERFITTING                                │
│                                                                          │
│  UNDERFITTING (Model quá đơn giản):                                     │
│  ─────────────────────────────────                                       │
│  • Train loss CAO, Val loss CAO                                          │
│  • Model chưa học được gì cả                                            │
│  • Fix: Train lâu hơn, LR cao hơn, model lớn hơn                       │
│                                                                          │
│  GOOD FIT (Lý tưởng):                                                   │
│  ────────────────────                                                    │
│  • Train loss THẤP, Val loss THẤP (gần nhau)                            │
│  • Model học tốt, generalize tốt                                        │
│  • Val loss ≤ 1.2× train loss → OK                                      │
│                                                                          │
│  OVERFITTING (Model "học thuộc lòng"):                                   │
│  ────────────────────────────────────                                    │
│  • Train loss RẤT THẤP, Val loss CAO (gap lớn)                          │
│  • Model nhớ training data, không generalize                            │
│  • Val loss > 1.5× train loss → có vấn đề                              │
│  • Fix: Early stopping, dropout, augmentation, ít epoch hơn             │
│                                                                          │
│  MINH HỌA (loss curve trên W&B):                                        │
│                                                                          │
│  Loss ↑                                                                  │
│  1.0 │ ╲                                                                │
│      │  ╲  ── val_loss (overfitting: bắt đầu tăng lại)                  │
│  0.5 │   ╲╱─────────────╲──────── val_loss (good fit)                   │
│      │    ╲              ╲                                               │
│  0.1 │     ╲──────────────── train_loss (luôn giảm)                     │
│      └──────────────────────────────► Epoch                              │
│       1    5    10   15   20                                             │
│                                                                          │
│  Early stopping sẽ dừng ở epoch val_loss bắt đầu tăng ↑                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: Đọc W&B Dashboard

Sau khi train, vào W&B dashboard (https://wandb.ai/hoangslevan-thu-dau-mot-university/holmhz) để xem:

```
┌─────────────────── W&B DASHBOARD ──────────────────────┐
│                                                         │
│  📊 Charts tab (quan trọng nhất):                       │
│  ───────────────────────────────                        │
│  1. train_loss / val_loss theo epoch                    │
│     → 2 đường phải gần nhau (no overfitting)            │
│     → Cả 2 phải giảm dần                               │
│                                                         │
│  2. val_auc theo epoch                                  │
│     → Phải tăng dần, rồi plateau (ổn định)             │
│     → Giá trị cuối = kết quả chính                     │
│                                                         │
│  3. lr theo epoch                                       │
│     → CosineAnnealing: giảm từ LR_max → gần 0          │
│     → Hình cosine (giảm nhanh đầu, chậm cuối)          │
│                                                         │
│  📋 Table tab:                                          │
│  ──────────                                             │
│  • So sánh nhiều runs (Phase 1, Phase 2, HP tuning)     │
│  • Sort theo val_auc để tìm run tốt nhất               │
│                                                         │
│  🔧 Config:                                             │
│  ──────────                                             │
│  • Hyperparameters dùng cho mỗi run                     │
│  • Dùng để reproduce kết quả                            │
│                                                         │
│  Mẹo: Chọn nhiều runs → overlay charts để so sánh      │
└─────────────────────────────────────────────────────────┘
```

---

## Tổng quan các bước

```
Bước 0:  Git branch ────────────────────── 5 phút
Bước 1:  Dọn dẹp dry run cũ ───────────── 2 phút
Bước 2:  Phase 1: Freeze train (10 ep) ── 20 min (RTX 3050)
Bước 3:  Phân tích Phase 1 ────────────── 15 phút
Bước 4:  Phase 2: Unfreeze train (20 ep)─ 3 giờ (RTX 3050)
Bước 5:  HP tuning (3 runs) ──────────── 6-9 giờ
Bước 6:  Chọn best model ──────────────── 30 phút
Bước 7:  Implement predict.py ─────────── 30 phút
Bước 8:  Smoke test imgs/ ─────────────── 15 phút
Bước 9:  Document results ─────────────── 30 phút
Bước 10: Commit & PR ─────────────────── 15 phút
                                    Tổng: ~3-4 ngày
```

> **Lưu ý**: Phase 2 + HP tuning tốn nhiều giờ GPU. Có thể chạy ban đêm.
> Nếu dùng Kaggle T4: nhanh hơn ~2×, nhưng cần setup notebook.

---

## Bước 0: Chuẩn bị Git branch

```bash
# Chuyển về main, pull latest
git checkout main
git pull origin main

# Tạo branch mới cho baseline training
git checkout -b feat/s1/baseline-training

# Verify branch
git branch
# * feat/s1/baseline-training
```

> **Tại sao branch riêng?**
> Phase 2 cần modify `train.py` (thêm tính năng load weights).
> Branch riêng = dễ revert nếu cần, PR review rõ ràng.

---

## Bước 1: Dọn dẹp dry run cũ

Dry run từ Task 1.5 để lại checkpoint cũ. Cần dọn để Phase 1 train từ đầu:

```bash
# Xóa checkpoint cũ (từ dry run Task 1.5)
rm -f outputs/checkpoints/best.pt outputs/checkpoints/last.pt

# Verify
ls outputs/checkpoints/
# (empty — OK)
```

> **Tại sao xóa?**
> `train.py` auto-resume từ `last.pt` nếu tồn tại.
> Dry run chỉ train 2 epoch → checkpoint không hữu ích cho Phase 1 (10 epoch).
> Xóa đi để train fresh.

Dọn luôn wandb local runs (optional, không ảnh hưởng kết quả):

```bash
# Optional: xóa wandb local runs cũ
rm -rf wandb/
```

---

## Bước 2: Phase 1 — Freeze backbone, train head

### Mục tiêu

- Train **chỉ head** (1,281 params) trong 10 epochs
- Backbone ĐÓng BĂNG — sử dụng pretrained ImageNet features
- Target: **Val AUC ≥ 0.90**

### Chạy trên Local (RTX 3050, 4GB VRAM)

```bash
# Phase 1: Freeze backbone (default config)
# batch_size=8 vì RTX 3050 chỉ 4GB VRAM
# data.num_workers=0 để tránh multiprocessing issues trên Windows
python scripts/train.py \
    training.epochs=10 \
    training.batch_size=8 \
    data.num_workers=0
```

> **Giải thích CLI:**
>
> - `training.epochs=10` → Train 10 epochs (thay default 30)
> - `training.batch_size=8` → 8 ảnh/batch (fit 4GB VRAM)
> - `data.num_workers=0` → Không dùng multiprocessing (Windows safe)
> - Config default: `freeze_backbone=true`, `lr=0.001`, `scheduler=cosine`
> - W&B: tự động log (`.env` chứa API key)

### Chạy trên Kaggle T4 (nhanh hơn ~2×)

```bash
# Kaggle: batch lớn hơn, worker nhiều hơn
python scripts/train.py \
    training.epochs=10 \
    training.batch_size=32 \
    data.num_workers=4
```

### Kết quả kỳ vọng (dựa trên dry run)

```
Epoch  1 | Train Loss: 0.52 | Val Loss: 0.40 | Val AUC: 0.91 | ~117s
Epoch  2 | Train Loss: 0.49 | Val Loss: 0.39 | Val AUC: 0.92 |
Epoch  3 | Train Loss: 0.47 | Val Loss: 0.38 | Val AUC: 0.93 |
...
Epoch 10 | Train Loss: 0.40 | Val Loss: 0.35 | Val AUC: 0.94-0.95 |

TRAINING COMPLETE — Best val_auc: ~0.94
```

> Checkpoint tự động lưu:
>
> - `outputs/checkpoints/best.pt` — model tốt nhất (val_auc cao nhất)
> - `outputs/checkpoints/last.pt` — model cuối cùng

### Nếu bị disconnect (Colab/Kaggle)

```bash
# Chạy lại cùng lệnh — tự động resume từ last.pt
python scripts/train.py \
    training.epochs=10 \
    training.batch_size=8 \
    data.num_workers=0

# Output sẽ hiện:
# Loading checkpoint: outputs/checkpoints/last.pt
# Resumed from epoch 5, best_metric=0.9350
# Epochs: 6 → 10 (tiếp tục từ epoch 6)
```

---

## Bước 3: Phân tích kết quả Phase 1

Sau khi Phase 1 xong, kiểm tra kết quả:

### 3.1 Xem W&B Dashboard

1. Mở browser: https://wandb.ai/hoangslevan-thu-dau-mot-university/holmhz
2. Click vào run mới nhất
3. Xem Charts:
   - `val_auc` → phải ≥ 0.90 (target Phase 1)
   - `train_loss` vs `val_loss` → gap nhỏ (no overfitting)
   - `lr` → cosine decay curve

### 3.2 Checklist Phase 1

```
[ ] Val AUC ≥ 0.90?
    ✅ → Tiếp tục Phase 2
    ❌ → Xem Troubleshooting

[ ] Val loss ≤ 1.2× train loss? (no overfitting)
    ✅ → Good fit
    ❌ → Tăng dropout hoặc giảm epochs

[ ] best.pt và last.pt tồn tại?
    ✅ → Sẵn sàng Phase 2
    ❌ → Checkpoint bị lỗi, xem lỗi trong logs

[ ] W&B dashboard có data?
    ✅ → Tiếp tục
    ❌ → Kiểm tra .env có WANDB_API_KEY
```

### 3.3 Sao lưu Phase 1 checkpoint

```bash
# Copy best.pt của Phase 1 sang tên riêng
cp outputs/checkpoints/best.pt outputs/checkpoints/phase1_best.pt

# Ghi chú AUC
echo "Phase 1 Best AUC: <thay_bằng_giá_trị_thực>" > outputs/checkpoints/phase1_notes.txt
```

---

## Bước 4: Phase 2 — Unfreeze, fine-tune toàn bộ

### Mục tiêu

- Train **toàn bộ** 4,008,829 params (backbone + head)
- LR thấp hơn 10× (1e-4) để không phá pretrained features
- Target: **Val AUC ≥ 0.93**

### 4.1 Chuẩn bị (QUAN TRỌNG!)

```bash
# ⚠️ PHẢI XÓA last.pt — tránh auto-resume với optimizer cũ
# Optimizer cũ chỉ optimize 1,281 head params
# Phase 2 cần optimizer mới cho TOÀN BỘ 4M params
rm -f outputs/checkpoints/last.pt outputs/checkpoints/best.pt

# Verify Phase 1 backup còn
ls outputs/checkpoints/phase1_best.pt
# phase1_best.pt — OK
```

> **⚠️ Tại sao PHẢI xóa last.pt?**
>
> `train.py` auto-resume từ `last.pt` nếu tồn tại.
> Nhưng Phase 1 optimizer chỉ có head params (1,281).
> Phase 2 cần optimizer cho toàn bộ params (4M).
>
> Nếu resume → optimizer cũ không biết backbone params → train sai!
>
> **DO**: Xóa checkpoint → train.py tạo mới optimizer cho toàn bộ params.
> **DON'T**: Để last.pt và chạy Phase 2 → optimizer mismatch.

### 4.2 Chạy Phase 2

```bash
# Phase 2: Unfreeze backbone + fine-tune
python scripts/train.py \
    model.freeze_backbone=false \
    training.learning_rate=0.0001 \
    training.epochs=20 \
    training.batch_size=8 \
    data.num_workers=0
```

> **Giải thích:**
>
> - `model.freeze_backbone=false` → Unfreeze toàn bộ backbone
> - `training.learning_rate=0.0001` → LR thấp hơn 10× (1e-4 thay 1e-3)
> - `training.epochs=20` → Train 20 epochs (nhiều hơn vì backbone cần thời gian)
> - `pretrained=true` vẫn được giữ → model start với ImageNet weights

> **Thời gian ước tính:**
>
> | GPU                  | Per epoch | 20 epochs |
> | -------------------- | --------- | --------- |
> | RTX 3050 (batch=8)   | ~8-10 min | ~3h       |
> | Kaggle T4 (batch=32) | ~4 min    | ~1.3h     |

### 4.3 Kết quả kỳ vọng

```
Epoch  1 | Train Loss: 0.45 | Val Loss: 0.35 | Val AUC: 0.93 | ~500s
Epoch  2 | Train Loss: 0.38 | Val Loss: 0.30 | Val AUC: 0.94 |
...
Epoch 10 | Train Loss: 0.20 | Val Loss: 0.22 | Val AUC: 0.95-0.96 |
...
Early stopping at epoch 15 (no improvement for 5 epochs)

TRAINING COMPLETE — Best val_auc: ~0.95
```

### 4.4 Xem W&B so sánh Phase 1 vs Phase 2

1. Mở W&B Dashboard
2. Chọn cả 2 runs (Phase 1 + Phase 2) → tick checkbox
3. So sánh `val_auc` curve: Phase 2 phải cao hơn Phase 1

---

## Bước 5: Hyperparameter tuning (Phase 2)

### Mục đích

Thử 3 giá trị LR cho Phase 2 để tìm LR tối ưu:

```
Run A: LR = 5e-4  (0.0005) — "bước lớn hơn"
Run B: LR = 1e-4  (0.0001) — "default" (đã chạy ở Bước 4)
Run C: LR = 5e-5  (0.00005) — "bước nhỏ nhất"
```

### 5.1 Run A: LR = 5e-4

```bash
# Xóa checkpoint cũ
rm -f outputs/checkpoints/last.pt outputs/checkpoints/best.pt

# Run A
python scripts/train.py \
    model.freeze_backbone=false \
    training.learning_rate=0.0005 \
    training.epochs=20 \
    training.batch_size=8 \
    data.num_workers=0

# Sau khi xong, backup best
cp outputs/checkpoints/best.pt outputs/checkpoints/hp_lr5e4_best.pt
```

### 5.2 Run C: LR = 5e-5

```bash
# Xóa checkpoint cũ
rm -f outputs/checkpoints/last.pt outputs/checkpoints/best.pt

# Run C
python scripts/train.py \
    model.freeze_backbone=false \
    training.learning_rate=0.00005 \
    training.epochs=20 \
    training.batch_size=8 \
    data.num_workers=0

# Sau khi xong, backup best
cp outputs/checkpoints/best.pt outputs/checkpoints/hp_lr5e5_best.pt
```

> **Run B (LR=1e-4) đã chạy ở Bước 4** → backup:
>
> ```bash
> cp outputs/checkpoints/phase2_best.pt outputs/checkpoints/hp_lr1e4_best.pt
> ```
>
> (Nếu bạn đã sao lưu Phase 2 checkpoint ở Bước 4)

### 5.3 So sánh trên W&B

```
┌──────────────── W&B COMPARISON ─────────────────┐
│                                                   │
│  1. Mở Dashboard → chọn 3 runs HP tuning          │
│  2. Charts → overlay val_auc                      │
│  3. Table → sort by val_auc DESC                  │
│                                                   │
│  Kỳ vọng:                                        │
│                                                   │
│  LR = 5e-4: AUC converge nhanh nhưng có thể      │
│             overshoot (không ổn định cuối)         │
│                                                   │
│  LR = 1e-4: AUC ổn định, converge tốt            │
│             (thường là sweet spot)                 │
│                                                   │
│  LR = 5e-5: AUC converge chậm, có thể chưa       │
│             đạt peak trong 20 epochs               │
│                                                   │
│  → Chọn run có val_auc CAO NHẤT và KHÔNG bị       │
│     overfitting (val_loss ≤ 1.2× train_loss)      │
└───────────────────────────────────────────────────┘
```

> **Mẹo tiết kiệm thời gian**: Nếu Bước 4 (LR=1e-4) đã cho AUC ≥ 0.93 và
> bạn thiếu GPU time → có thể skip HP tuning. Ghi chú "chỉ test 1 LR" trong
> báo cáo, nói đã đạt target. Hội đồng quan tâm AUC ≥ 0.85 hơn là HP tuning.

---

## Bước 6: Chọn best model + phân tích

### 6.1 So sánh tất cả runs

Tạo bảng tổng kết:

| Run               | Phase     | freeze_backbone | LR   | Epochs | Best Val AUC | Val Loss | Overfitting? |
| ----------------- | --------- | --------------- | ---- | ------ | ------------ | -------- | ------------ |
| Phase 1           | Freeze    | true            | 1e-3 | 10     | ?            | ?        | ?            |
| Phase 2 (LR=5e-4) | Fine-tune | false           | 5e-4 | 20     | ?            | ?        | ?            |
| Phase 2 (LR=1e-4) | Fine-tune | false           | 1e-4 | 20     | ?            | ?        | ?            |
| Phase 2 (LR=5e-5) | Fine-tune | false           | 5e-5 | 20     | ?            | ?        | ?            |

### 6.2 Chọn final best checkpoint

```bash
# Chọn best checkpoint (ví dụ LR=1e-4 cho kết quả tốt nhất)
# Copy làm "official" best checkpoint
cp outputs/checkpoints/hp_lr1e4_best.pt outputs/checkpoints/best.pt

# Kiểm tra size
ls -lh outputs/checkpoints/best.pt
# ~16MB — OK
```

### 6.3 Kiểm tra tiêu chí Milestone 1

```
┌─────────────── MILESTONE 1 CHECKLIST ──────────────┐
│                                                       │
│  [ ] Dataset ≥ 15K ảnh                               │
│      → 18,550 train + 3,975 val = 22,525 ✅          │
│                                                       │
│  [ ] Baseline AUC ≥ 0.85 (in-domain val set)         │
│      → Best AUC = ??? (điền giá trị thực)            │
│                                                       │
│  [ ] W&B dashboard có training curves                 │
│      → Link: https://wandb.ai/...                    │
│                                                       │
│  [ ] Checkpoint saved                                 │
│      → outputs/checkpoints/best.pt                   │
│                                                       │
│  Nếu TẤT CẢ ✅ → Proceed to Sprint 2!               │
└───────────────────────────────────────────────────────┘
```

---

## Bước 7: Implement predict.py (smoke test)

File `scripts/predict.py` hiện EMPTY. Cần implement để chạy inference trên ảnh đơn lẻ:

### 7.1 Tạo predict.py

```python
"""
HolmHz Prediction Script — Inference trên 1 hoặc nhiều ảnh.

Usage:
    # Predict 1 ảnh
    python scripts/predict.py imgs/Real/IMG_2344.jpg

    # Predict cả folder
    python scripts/predict.py imgs/Fake_AI_generated/

    # Custom checkpoint
    python scripts/predict.py imgs/Real/IMG_2344.jpg --checkpoint outputs/checkpoints/best.pt
"""

import sys
from pathlib import Path

import cv2
import torch
from dotenv import load_dotenv

load_dotenv()

import holmhz.detectors  # noqa: E402, F401
from holmhz.data.transforms import get_val_transforms
from holmhz.utils.logger import get_logger
from holmhz.utils.registry import DETECTOR_REGISTRY

logger = get_logger("predict")


def predict_single(
    image_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> dict:
    """Predict 1 ảnh.

    Returns:
        dict: {"path", "prob_fake", "label", "confidence"}
    """
    # Load ảnh
    img = cv2.imread(image_path)
    if img is None:
        return {"path": image_path, "error": "Cannot load image"}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform
    transformed = transform(image=img)
    tensor = transformed["image"].unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor)                    # [1, 1]
        prob = torch.sigmoid(logits).item()       # P(Fake) ∈ [0, 1]

    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else 1 - prob

    return {
        "path": image_path,
        "prob_fake": round(prob, 4),
        "label": label,
        "confidence": round(confidence, 4),
    }


def main():
    """Main prediction entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <image_or_folder> [--checkpoint path]")
        sys.exit(1)

    input_path = sys.argv[1]
    checkpoint_path = "outputs/checkpoints/best.pt"

    # Parse --checkpoint arg
    if "--checkpoint" in sys.argv:
        idx = sys.argv.index("--checkpoint")
        checkpoint_path = sys.argv[idx + 1]

    # Check checkpoint exists
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Train a model first: python scripts/train.py")
        sys.exit(1)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model = DETECTOR_REGISTRY.build(
        "efficientnet_b0",
        pretrained=False,   # Sẽ load weights từ checkpoint
        dropout=0.3,
        freeze_backbone=False,
    )

    # Load checkpoint (chỉ model weights)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    best_auc = checkpoint.get("best_metric", "N/A")
    epoch = checkpoint.get("epoch", "N/A")
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch}, AUC {best_auc})")

    # Transform (val transform — no augmentation)
    transform = get_val_transforms(image_size=224)

    # Collect image paths
    input_p = Path(input_path)
    if input_p.is_file():
        image_paths = [str(input_p)]
    elif input_p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            str(f) for f in input_p.rglob("*") if f.suffix.lower() in exts
        )
    else:
        logger.error(f"Path not found: {input_path}")
        sys.exit(1)

    if not image_paths:
        logger.error(f"No images found in: {input_path}")
        sys.exit(1)

    logger.info(f"Predicting {len(image_paths)} images...")

    # Predict
    results = []
    print("\n" + "=" * 70)
    print(f"{'Path':<50} {'Label':<6} {'P(Fake)':<8} {'Conf':<6}")
    print("=" * 70)

    for img_path in image_paths:
        result = predict_single(img_path, model, transform, device)
        results.append(result)

        if "error" in result:
            print(f"{Path(img_path).name:<50} ERROR: {result['error']}")
        else:
            name = Path(img_path).name
            if len(name) > 48:
                name = name[:45] + "..."
            print(
                f"{name:<50} "
                f"{result['label']:<6} "
                f"{result['prob_fake']:<8.4f} "
                f"{result['confidence']:<6.4f}"
            )

    print("=" * 70)

    # Summary
    valid = [r for r in results if "error" not in r]
    fakes = sum(1 for r in valid if r["label"] == "FAKE")
    reals = sum(1 for r in valid if r["label"] == "REAL")
    print(f"\nSummary: {len(valid)} images — {fakes} FAKE, {reals} REAL")


if __name__ == "__main__":
    main()
```

### 7.2 Ghi code vào file

Copy toàn bộ code trên vào `scripts/predict.py`.

> **Giải thích flow:**
>
> 1. Load model architecture (EfficientNetDetector)
> 2. Load trained weights từ `best.pt` checkpoint
> 3. Load ảnh → val_transform (resize + normalize, no augmentation)
> 4. Forward pass → logits → sigmoid → P(Fake)
> 5. P(Fake) ≥ 0.5 → "FAKE", ngược lại → "REAL"

---

## Bước 8: Smoke test trên imgs/

### 8.1 Chạy predict trên folder

```bash
# Test trên Fake images (5 ảnh)
python scripts/predict.py imgs/Fake_AI_generated/

# Test trên Real images (5 ảnh)
python scripts/predict.py imgs/Real/

# Hoặc test cả 2 cùng lúc
python scripts/predict.py imgs/
```

### 8.2 Kết quả kỳ vọng

```
======================================================================
Path                                               Label  P(Fake)  Conf
======================================================================
Gemini_Generated_Image_h2x4b6h2x4b6h2x4.png       FAKE   0.9234   0.9234
generation-07888628-525f-4796-959d-c25685368055.png FAKE   0.8765   0.8765
generation-9f6e6243-fe40-45b3-bc62-88aa8183de1d.png FAKE   0.7891   0.7891
generation-d69056e8-656c-422f-a0a9-679c5333361d.png FAKE   0.8432   0.8432
generation-e9eac301-df04-4180-a36a-fcc399ebda7c.png FAKE   0.9012   0.9012
======================================================================

Summary: 5 images — 5 FAKE, 0 REAL
```

```
======================================================================
Path                                               Label  P(Fake)  Conf
======================================================================
IMG_20211207_152750_319.jpg                        REAL   0.1234   0.8766
IMG_20211207_153512_056.jpg                        REAL   0.0987   0.9013
IMG_2344.jpg                                       REAL   0.2345   0.7655
IMG_2365.jpg                                       REAL   0.1876   0.8124
IMG_2369.jpg                                       REAL   0.0654   0.9346
======================================================================

Summary: 5 images — 0 FAKE, 5 REAL
```

### 8.3 Acceptance Criteria

```
[ ] ≥ 3/5 Fake images detected as FAKE    (P(Fake) ≥ 0.5)
[ ] ≥ 3/5 Real images detected as REAL    (P(Fake) < 0.5)

⚠️ Lưu ý: Ảnh từ imgs/ là Gemini/Flux — OOD (chưa có trong training data!)
Model có thể dự đoán sai 1-2 ảnh OOD → vẫn OK cho baseline.
Accuracy trên OOD sẽ được đánh giá chính thức ở Task 2.1.
```

---

## Bước 9: Document results (CONTEXT.md)

Thêm vào `docs/CONTEXT.md` section mới:

```markdown
## 14. Baseline Training Progress (Task 1.6) — ✅ COMPLETED DD/MM/2026

### Tổng quan

- **Mục tiêu**: Train EfficientNet-B0 baseline trên 18,550 ảnh (GAN + Diffusion)
- **Branch**: `feat/s1/baseline-training`
- **Strategy**: Phase 1 (freeze backbone) → Phase 2 (fine-tune toàn bộ)
- **Best Val AUC**: ??? (Phase 2, LR=???)
- **W&B**: https://wandb.ai/hoangslevan-thu-dau-mot-university/holmhz

### Phase 1: Freeze backbone (head only)

| Config           | Value |
| ---------------- | ----- |
| freeze_backbone  | true  |
| trainable params | 1,281 |
| LR               | 1e-3  |
| Epochs           | 10    |
| Batch size       | 8     |

| Epoch | Train Loss | Val Loss | Val Acc | Val AUC |
| ----- | ---------- | -------- | ------- | ------- |
| 1     | ?          | ?        | ?       | ?       |
| ...   | ...        | ...      | ...     | ...     |
| 10    | ?          | ?        | ?       | ?       |

### Phase 2: Fine-tune (unfreeze)

| Config           | Value     |
| ---------------- | --------- |
| freeze_backbone  | false     |
| trainable params | 4,008,829 |
| LR               | 1e-4      |
| Epochs           | 20        |
| Batch size       | 8         |

Best Val AUC: ???

### HP Tuning Results

| LR   | Best Val AUC | Epochs run | Overfitting? |
| ---- | ------------ | ---------- | ------------ |
| 5e-4 | ?            | ?          | ?            |
| 1e-4 | ?            | ?          | ?            |
| 5e-5 | ?            | ?          | ?            |

### Smoke Test (imgs/)

- Fake: ?/5 detected correctly
- Real: ?/5 detected correctly

### Milestone 1 Status

- [x] Dataset ≥ 15K: 22,525 ✅
- [ ] Baseline AUC ≥ 0.85: ??? (target: ≥ 0.93)
```

> **Lưu ý**: Điền giá trị thực tế sau khi train xong. Copilot sẽ giúp format.

---

## Bước 10: Commit & PR

```bash
# Stage changes
git add scripts/predict.py
git add docs/CONTEXT.md
git add docs/tasks/TASK_1.6_BASELINE_TRAINING.md
git add -A  # Any other changes

# Commit
git commit -m "feat(s1): baseline training — Phase 1+2 complete

- Phase 1: freeze backbone, 10 epochs → Val AUC ???
- Phase 2: fine-tune, 20 epochs, LR=1e-4 → Val AUC ???
- HP tuning: tested LR {5e-4, 1e-4, 5e-5}
- Implement scripts/predict.py for smoke test
- Smoke test: ?/5 Fake, ?/5 Real correct
- W&B dashboard: all runs logged
- Milestone 1: AUC ≥ 0.85 ✅"

# Push
git push -u origin feat/s1/baseline-training
```

> **PR Description nên include:**
>
> - W&B dashboard link
> - Bảng so sánh Phase 1 vs Phase 2 AUC
> - Screenshot training curves
> - Smoke test results

---

## Checklist hoàn thành

```
TASK 1.6 — BASELINE TRAINING

Subtask 1.6.1: Phase 1 — Freeze backbone
  [ ] 10 epochs completed
  [ ] Val AUC ≥ 0.90
  [ ] W&B run logged
  [ ] phase1_best.pt saved

Subtask 1.6.2: Phase 2 — Fine-tune
  [ ] 20 epochs completed (hoặc early stopped)
  [ ] Val AUC ≥ 0.93
  [ ] W&B run logged
  [ ] best.pt saved

Subtask 1.6.3: HP Tuning
  [ ] ≥ 3 LR values tested
  [ ] W&B comparison chart
  [ ] Best LR documented

Subtask 1.6.4: Smoke test
  [ ] scripts/predict.py implemented
  [ ] ≥ 3/5 Fake detected correctly
  [ ] ≥ 3/5 Real detected correctly

Subtask 1.6.5: Documentation
  [ ] CONTEXT.md updated with results
  [ ] Training time documented
  [ ] W&B link documented

Branch & PR
  [ ] Branch: feat/s1/baseline-training
  [ ] All changes committed
  [ ] PR created with W&B link

Milestone 1
  [ ] Dataset ≥ 15K ✅ (22,525)
  [ ] Baseline AUC ≥ 0.85
```

---

## Troubleshooting

### Phase 1 AUC < 0.90 sau 10 epochs

```
Nguyên nhân: Có thể CosineAnnealing LR decay quá nhanh
(LR giảm xuống ~0 ở epoch 10)

Fix options:
1. Tăng epochs: training.epochs=20
2. Tăng LR: training.learning_rate=0.003
3. Kiểm tra data balance: train.json có ~50/50 real/fake?
   → python -c "import json; d=json.load(open('data/manifests/train.json'));
     print(sum(1 for x in d if x['label']==1), '/', len(d))"
```

### Phase 2 AUC GIẢM so với Phase 1

```
Nguyên nhân: Learning rate quá cao khi unfreeze
→ Catastrophic forgetting (model quên ImageNet knowledge)

Fix:
1. Giảm LR: training.learning_rate=0.00005 (5e-5)
2. Hoặc thử 1e-5 (rất nhỏ)
3. Nếu vẫn giảm → dùng Phase 1 checkpoint, skip Phase 2
   (Ghi chú: "Phase 1 freeze đã đủ tốt cho baseline")
```

### Out of Memory (OOM) trên RTX 3050

```
Triệu chứng: CUDA out of memory

Fix:
1. Giảm batch: training.batch_size=4
2. Nếu vẫn OOM: training.batch_size=2
3. Nếu vẫn OOM: dùng CPU (chậm 10×)
   → python scripts/train.py training.batch_size=16 data.num_workers=0
   (Không chỉ định device → auto-detect, nếu CUDA OOM PyTorch sẽ error)
```

### Out of Memory khi Unfreeze (Phase 2)

```
Unfreeze = train 4M params → cần nhiều VRAM hơn Phase 1

Fix:
1. Giảm batch: training.batch_size=4 (thay 8)
2. AMP đã bật sẵn (mixed precision fp16) → tiết kiệm ~40% VRAM
3. Nếu vẫn OOM → freeze 1 số early layers (advanced):
   # Freeze các block đầu (layers 0-3), chỉ unfreeze block 4-7 + head
   # Cần modify code — hỏi Copilot nếu cần
```

### W&B không log (run offline)

```
Triệu chứng: "W&B disabled" hoặc không thấy run trên dashboard

Fix:
1. Kiểm tra .env: cat .env | grep WANDB
2. Test: python -c "import wandb; wandb.login()"
3. Nếu mạng yếu → dùng offline mode:
   WANDB_MODE=offline python scripts/train.py ...
   # Sau khi xong:
   wandb sync wandb/latest-run/
```

### Training quá chậm

```
RTX 3050 batch=8: ~2 min/epoch (freeze), ~8 min/epoch (unfreeze)

Nếu chậm hơn nhiều:
1. Kiểm tra GPU có đang dùng: nvidia-smi
2. data.num_workers=0 chậm hơn =4 nhưng an toàn trên Windows
3. Tắt các app nặng (browser, VS Code Extensions)
4. Cân nhắc Kaggle T4 (nhanh gấp 2-3×)
```

### Checkpoint resume bị lỗi

```
Triệu chứng: Error khi load_checkpoint

Fix:
1. Xóa checkpoint: rm outputs/checkpoints/last.pt
2. Train lại từ đầu
3. Nếu lỗi "key mismatch": model architecture đã thay đổi
   → Xóa cả best.pt, train lại hoàn toàn
```

---

## Mối liên hệ với các Task tiếp theo

```
┌──────────────── SAU KHI TASK 1.6 XONG ────────────────┐
│                                                         │
│  outputs/checkpoints/best.pt (model tốt nhất)           │
│    │                                                     │
│    ├──► Task 2.1: Evaluation Pipeline                    │
│    │    • Chạy best.pt trên test_id (3,975 ảnh)          │
│    │    • Chạy trên test_ood (1,180 ảnh: Flux, etc.)     │
│    │    • Per-source accuracy breakdown                   │
│    │    • ROC curve + Confusion matrix                    │
│    │                                                     │
│    ├──► Task 2.2: Benchmark SOTA                         │
│    │    • So sánh HolmHz vs CNNDetection vs UFD vs DFB   │
│    │    • Chạy 3 SOTA trên CÙNG test set                 │
│    │    • Tạo bảng comparison cho báo cáo                 │
│    │                                                     │
│    ├──► Task 2.3: Grad-CAM XAI                           │
│    │    • model.get_feature_layer() → conv_head           │
│    │    • Tạo heatmap giải thích model nhìn vùng nào      │
│    │    • "Tại sao model nói ảnh này là Fake?"            │
│    │                                                     │
│    └──► Task 2.4: Model Export (ONNX)                    │
│         • Export best.pt → model.onnx                    │
│         • Optimize cho inference nhanh (web demo)         │
│                                                         │
│  scripts/predict.py                                      │
│    └──► Task 3.1: Backend API                            │
│         • FastAPI wraps predict logic                     │
│         • POST /api/predict → JSON response              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

> **Quan trọng nhất**: `best.pt` từ Task 1.6 sẽ được dùng xuyên suốt Sprint 2-4.
> Nếu AUC thấp → quay lại thêm data (FFHQ 52K backup) hoặc thay đổi augmentation.
> Nếu AUC ≥ 0.90 → tự tin proceed, kết quả đủ tốt cho báo cáo tốt nghiệp.
