# 📖 HƯỚNG DẪN CHI TIẾT TASK 1.5: TRAINING PIPELINE

> **Dành cho**: Lê Văn Hoàng — người chưa có nền tảng ML/DL, học qua thực hành  
> **Triết lý**: Mỗi bước không chỉ hướng dẫn **làm gì** mà giải thích **tại sao làm vậy**  
> **Thời gian**: ~4-5 ngày  
> **Tiền đề**: Task 1.3 Data Pipeline ✅ | Task 1.4 Model Architecture ✅  
> **Tham chiếu**: [TASK_1.5_TRAINING_PIPELINE.md](../tasks/TASK_1.5_TRAINING_PIPELINE.md) | [PROJECT_PLAN.md](../PROJECT_PLAN.md) Section 6
>
> **Output**: Training pipeline hoàn chỉnh — chạy `python scripts/train.py` để train model end-to-end

---

## 📋 Mục lục

- [Bức tranh tổng thể: Training Pipeline nằm ở đâu?](#bức-tranh-tổng-thể-training-pipeline-nằm-ở-đâu)
- [Mối quan hệ với Task 1.3 + 1.4](#mối-quan-hệ-với-task-13--14)
- [Kiến thức nền: Training Loop](#kiến-thức-nền-training-loop)
- [Kiến thức nền: Loss Function (BCEWithLogitsLoss)](#kiến-thức-nền-loss-function-bcewithlogitsloss)
- [Kiến thức nền: Optimizer (AdamW)](#kiến-thức-nền-optimizer-adamw)
- [Kiến thức nền: Learning Rate Scheduler](#kiến-thức-nền-learning-rate-scheduler)
- [Kiến thức nền: Early Stopping](#kiến-thức-nền-early-stopping)
- [Kiến thức nền: Checkpoint & Resume](#kiến-thức-nền-checkpoint--resume)
- [Kiến thức nền: Mixed Precision (AMP)](#kiến-thức-nền-mixed-precision-amp)
- [Kiến thức nền: W&B Experiment Tracking](#kiến-thức-nền-wb-experiment-tracking)
- [Tổng quan các bước](#tổng-quan-các-bước)
- [Bước 0: Chuẩn bị Git branch](#bước-0-chuẩn-bị-git-branch)
- [Bước 1: Implement Metrics (accuracy + AUC)](#bước-1-implement-metrics-accuracy--auc)
- [Bước 2: Implement Loss Function](#bước-2-implement-loss-function)
- [Bước 3: Implement Logger](#bước-3-implement-logger)
- [Bước 4: Implement LR Scheduler](#bước-4-implement-lr-scheduler)
- [Bước 5: Implement Early Stopping](#bước-5-implement-early-stopping)
- [Bước 6: Implement Trainer](#bước-6-implement-trainer)
- [Bước 7: Update **init**.py exports](#bước-7-update-__init__py-exports)
- [Bước 8: Implement scripts/train.py](#bước-8-implement-scriptstrainpy)
- [Bước 9: Unit tests](#bước-9-unit-tests)
- [Bước 10: Dry run (2 epochs, 100 ảnh)](#bước-10-dry-run-2-epochs-100-ảnh)
- [Bước 11: Commit & PR](#bước-11-commit--pr)
- [Checklist hoàn thành](#checklist-hoàn-thành)
- [Troubleshooting](#troubleshooting)
- [Mối liên hệ với các Task tiếp theo](#mối-liên-hệ-với-các-task-tiếp-theo)

---

## Bức tranh tổng thể: Training Pipeline nằm ở đâu?

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        DỰ ÁN HOLMHZ — SPRINT 1                          │
│                                                                           │
│  Task 1.1  Setup môi trường ✅ DONE                                      │
│  Task 1.2  Thu thập dữ liệu ✅ DONE (27,680 ảnh)                        │
│  Task 1.3  Data Pipeline    ✅ DONE (17/17 tests pass)                   │
│    │                                                                      │
│    │  Output: DataLoader trả batch                                       │
│    │  {"image": [B,3,224,224], "label": [B], "source": [...]}            │
│    │                                                                      │
│  Task 1.4  Model Architecture ✅ DONE (30/30 tests pass)                 │
│    │                                                                      │
│    │  Output: EfficientNetDetector                                       │
│    │  model(x) → logits [B, 1]                                           │
│    │                                                                      │
│  ► Task 1.5  TRAINING PIPELINE  ◄◄◄  BẠN ĐANG Ở ĐÂY                    │
│    │                                                                      │
│    │  Đây là "công xưởng" ghép DATA + MODEL thành quy trình              │
│    │  tự động HỌC phân biệt Real vs Fake.                                │
│    │                                                                      │
│    │  7 thành phần:                                                       │
│    │    1. Metrics  — đo lường (accuracy, AUC)                           │
│    │    2. Loss     — hàm mất mát (BCEWithLogitsLoss)                    │
│    │    3. Optimizer — cập nhật weights (AdamW)                           │
│    │    4. Scheduler — điều chỉnh learning rate (CosineAnnealing)        │
│    │    5. Early Stopping — dừng sớm khi overfitting                     │
│    │    6. Trainer  — orchestrate toàn bộ training loop                   │
│    │    7. train.py — CLI entry point, đọc config YAML                   │
│    │                                                                      │
│    │  Assignee: Hoàng                                                     │
│    │  Target:   14/03/2026                                                │
│    │                                                                      │
│    └──► Task 1.6  Baseline Training                                      │
│              │                                                            │
│              │  Chạy train thực sự trên 18,550 ảnh (Kaggle GPU)          │
│              │  Evaluate AUC trên val + OOD test                         │
│              ▼                                                            │
│         Sprint 2: Evaluation + XAI + Benchmark                            │
│                                                                           │
│  ⚡ Task 1.3 + 1.4 đã xong → giờ chỉ tập trung vào 1.5                 │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Mối quan hệ với Task 1.3 + 1.4

Task 1.5 không tạo gì HOÀN TOÀN MỚI — nó **ghép** 2 phần đã có:

```python
# ══════════════════════════════════════════════════════════════
# TASK 1.3 đã tạo: DataLoader
# ══════════════════════════════════════════════════════════════
from holmhz.data import create_dataloader

train_loader = create_dataloader(
    manifest_path="data/manifests/train.json",
    batch_size=32,
    is_training=True,      # shuffle + augment
    num_workers=4,
)
# Trả về batch:
# {"image": [B, 3, 224, 224], "label": [B], "source": [...], "path": [...]}

# ══════════════════════════════════════════════════════════════
# TASK 1.4 đã tạo: Model
# ══════════════════════════════════════════════════════════════
from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors  # trigger registration

model = DETECTOR_REGISTRY.build(
    "efficientnet_b0",
    pretrained=True,
    dropout=0.3,
    freeze_backbone=True,
)
# model(images) → logits [B, 1]
# model.predict_proba(images) → probs [B, 1] ∈ [0, 1]

# ══════════════════════════════════════════════════════════════
# TASK 1.5 ghép thành: Training Loop
# ══════════════════════════════════════════════════════════════
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(30):
    for batch in train_loader:
        images = batch["image"].to(device)       # [B, 3, 224, 224]
        labels = batch["label"].to(device)       # [B]

        logits = model(images)                   # [B, 1]
        loss = loss_fn(logits.squeeze(1), labels) # squeeze [B,1] → [B]

        loss.backward()       # Tính gradient
        optimizer.step()      # Cập nhật weights
        optimizer.zero_grad() # Reset gradient
```

> **Lưu ý quan trọng**: `logits.squeeze(1)` — model trả về `[B, 1]` nhưng labels có shape `[B]`. Phải squeeze dimension 1 (không dùng `squeeze()` vì nếu B=1 sẽ bị mất hết dimensions).

---

## Kiến thức nền: Training Loop

### Epoch, Batch, Iteration — 3 đơn vị thời gian

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP — TỔNG QUAN                            │
│                                                                         │
│  Dataset: 18,550 ảnh train                                              │
│  Batch size: 32                                                         │
│  → 18,550 / 32 ≈ 579 batches = 579 ITERATIONS / epoch                  │
│  → 30 epochs = 30 × 579 = 17,370 iterations tổng                       │
│                                                                         │
│  ┌─────────────────────── EPOCH 1 ────────────────────────┐             │
│  │                                                         │             │
│  │  Iteration 1:   batch 32 ảnh → forward → loss → backward│             │
│  │  Iteration 2:   batch 32 ảnh → forward → loss → backward│             │
│  │  ...                                                    │             │
│  │  Iteration 579: batch cuối   → forward → loss → backward│             │
│  │                                                         │             │
│  │  → Validate trên 3,975 ảnh val                          │             │
│  │  → Log metrics (loss, accuracy, AUC)                    │             │
│  │  → Scheduler step (giảm LR)                             │             │
│  │  → Early stopping check                                 │             │
│  │  → Save checkpoint (best.pt nếu val_auc cải thiện)      │             │
│  │                                                         │             │
│  └─────────────────────────────────────────────────────────┘             │
│                                                                         │
│  ┌─────────────────────── EPOCH 2 ────────────────────────┐             │
│  │  (Lặp lại, data được shuffle lại mỗi epoch)            │             │
│  └─────────────────────────────────────────────────────────┘             │
│  ...                                                                    │
│  ┌─────────────────────── EPOCH 30 (hoặc sớm hơn) ───────┐             │
│  │  Early stopping: val_auc không tăng 5 epoch → DỪNG     │             │
│  └─────────────────────────────────────────────────────────┘             │
│                                                                         │
│  Output: best.pt (model tốt nhất) + last.pt (model cuối)               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Forward → Loss → Backward → Step (4 bước mỗi iteration)

```
┌──────────────────────────────────────────────────────────────────────────┐
│               1 ITERATION — 4 BƯỚC                                       │
│                                                                          │
│  BƯỚC 1: FORWARD PASS                                                    │
│  ─────────────────────                                                   │
│  images [32, 3, 224, 224] → model → logits [32, 1]                      │
│                                                                          │
│  BƯỚC 2: COMPUTE LOSS                                                    │
│  ────────────────────                                                    │
│  loss = BCEWithLogitsLoss(logits.squeeze(1), labels)                    │
│  → Một con số: loss = 0.6932 (cao = dự đoán sai nhiều)                 │
│                                                                          │
│  BƯỚC 3: BACKWARD PASS (Backpropagation)                                │
│  ──────────────────────                                                  │
│  loss.backward()                                                         │
│  → Tính gradient: "mỗi weight nên tăng/giảm bao nhiêu để loss giảm?"  │
│  → Gradient chạy ngược từ loss → head → backbone (nếu không freeze)    │
│                                                                          │
│  BƯỚC 4: OPTIMIZER STEP                                                  │
│  ────────────────────                                                    │
│  optimizer.step()      ← Cập nhật weights theo gradient                 │
│  optimizer.zero_grad() ← Reset gradient để iteration tiếp theo          │
│                                                                          │
│  TƯƠNG TỰ ĐỜI THỰC:                                                    │
│    Bước 1: Làm bài thi (forward)                                        │
│    Bước 2: Chấm điểm (loss — bao nhiêu sai?)                           │
│    Bước 3: Phân tích lỗi sai (backward — sai ở đâu?)                   │
│    Bước 4: Học lại phần sai (optimizer — sửa lỗi)                       │
│    → Lặp lại cho đến khi điểm đủ cao                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### model.train() vs model.eval() — 2 chế độ

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAIN vs EVAL MODE                                    │
│                                                                         │
│  model.train()                          model.eval()                    │
│  ──────────────                         ────────────                    │
│  Dropout: BẬT (tắt 30% neuron)         Dropout: TẮT (dùng hết)        │
│  BatchNorm: cập nhật running stats      BatchNorm: dùng running stats  │
│  Gradient: enabled                      Gradient: disabled (no_grad)   │
│                                                                         │
│  Dùng khi: TRAINING                     Dùng khi: VALIDATION, PREDICT  │
│                                                                         │
│  ⚠️ QUÊN gọi model.eval() khi validate → metrics sai (Dropout random) │
│  ⚠️ QUÊN gọi model.train() sau validate → training bị lỗi             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: Loss Function (BCEWithLogitsLoss)

Đã giải thích trong Guide Task 1.4. Tóm tắt lại:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BCEWithLogitsLoss                                     │
│                                                                         │
│  Model output: logits [B, 1]  (raw scores, ví dụ: [-2.1, 3.5, 0.2])   │
│  Labels:       [B]            (float32: 0.0 = Real, 1.0 = Fake)       │
│                                                                         │
│  Bên trong BCEWithLogitsLoss:                                           │
│    1. sigmoid(logits) → probs [0, 1]                                    │
│    2. BCE(probs, labels) → loss value                                   │
│    → Gộp 2 bước → numerical stability tốt hơn                          │
│                                                                         │
│  ⚠️ QUAN TRỌNG: logits shape [B, 1], labels shape [B]                  │
│  → Cần squeeze: loss_fn(logits.squeeze(1), labels)                     │
│                                                                         │
│  Loss interpretation:                                                    │
│    loss ≈ 0.693 → model đoán random (50/50)                            │
│    loss → 0.0   → model dự đoán gần đúng hết                           │
│    loss → ∞     → model dự đoán ngược hoàn toàn                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: Optimizer (AdamW)

### "Thợ sửa weights" — Optimizer làm gì?

Sau khi `loss.backward()` tính gradient, optimizer dùng gradient để **cập nhật weights**:

```
weight_mới = weight_cũ - learning_rate × gradient
```

Nếu gradient nói "weight này nên GIẢM 0.01" và LR = 0.001:

```
weight_mới = weight_cũ - 0.001 × 0.01 = weight_cũ - 0.00001
```

### Tại sao AdamW mà không phải SGD?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                SO SÁNH OPTIMIZER                                        │
│                                                                         │
│  SGD (Stochastic Gradient Descent):                                     │
│    weight -= lr × gradient                                              │
│    Đơn giản, nhưng nhạy cảm với LR, hội tụ chậm.                      │
│    Cần LR scheduling + momentum cẩn thận.                               │
│                                                                         │
│  Adam (Adaptive Moment Estimation):                                     │
│    • Tự điều chỉnh LR cho từng weight                                  │
│    • Dùng "momentum" (trung bình gradient gần đây)                     │
│    • Hội tụ nhanh hơn SGD, ít nhạy cảm với LR ban đầu                 │
│                                                                         │
│  AdamW (Adam + Weight Decay):                                           │
│    • Giống Adam + thêm weight decay (regularization)                   │
│    • Weight decay: "phạt" weight lớn → chống overfitting               │
│    • PyTorch mặc định tách weight decay ra khỏi gradient               │
│    • KHUYÊN DÙNG cho transfer learning (tốt hơn Adam)                  │
│                                                                         │
│  HolmHz dùng AdamW:                                                     │
│    lr = 0.001, weight_decay = 0.0001                                   │
│    → Hội tụ nhanh + regularization nhẹ                                 │
│    → Config: configs/train.yaml → training.optimizer: adamw            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Learning Rate — Tốc độ học

```
┌─────────────────────────────────────────────────────────────────────────┐
│                LEARNING RATE — QUAN TRỌNG NHẤT                          │
│                                                                         │
│  LR quá LỚN (0.1):   Bước nhảy quá xa → loss nhảy lung tung           │
│                        ████████████████████████                         │
│                        Model không hội tụ được                          │
│                                                                         │
│  LR vừa phải (0.001): Bước nhảy vừa đúng → loss giảm đều              │
│                        ████░░░░░░░░░░░░░░░░                             │
│                        ★ HolmHz dùng giá trị này                       │
│                                                                         │
│  LR quá NHỎ (0.00001): Bước nhảy quá bé → train rất lâu               │
│                         ████████████████░░░                             │
│                         Có thể kẹt ở local minimum                     │
│                                                                         │
│  → Bắt đầu lr=0.001, dùng scheduler giảm dần theo thời gian           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: Learning Rate Scheduler

### Tại sao giảm LR theo thời gian?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                COSINE ANNEALING LR SCHEDULER                            │
│                                                                         │
│  Đầu training: LR cao (0.001) → bước nhảy lớn, khám phá nhanh         │
│  Giữa training: LR giảm dần → bước nhảy nhỏ hơn, tinh chỉnh          │
│  Cuối training: LR rất nhỏ (1e-6) → bước nhảy tí hon, polish          │
│                                                                         │
│  LR                                                                     │
│  0.001 ┤  ★                                                            │
│        │    ╲                                                           │
│        │      ╲                                                         │
│        │        ╲                                                       │
│  0.0005┤          ╲                                                     │
│        │            ╲                                                   │
│        │              ╲╲                                                │
│        │                ╲╲╲                                             │
│  1e-6  ┤                   ╲╲╲╲______★                                 │
│        └────────────────────────────────                                │
│        Epoch 0          15          30                                  │
│                                                                         │
│  TƯƠNG TỰ: Lái xe đến gần đích                                        │
│    Xa đích → đạp ga mạnh (LR cao) → đến nhanh                         │
│    Gần đích → nhả ga (LR giảm) → đỗ chính xác                         │
│                                                                         │
│  Config: training.scheduler: cosine                                     │
│  Code: CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: Early Stopping

### "Biết dừng đúng lúc" — Chống overfitting

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EARLY STOPPING                                       │
│                                                                         │
│  Accuracy                                                                │
│  100% ┤                          ╱╱╱╱╱╱╱╱╱ ← Train accuracy (tăng mãi)│
│       │                     ╱╱╱╱╱                                       │
│   95% ┤                ╱╱╱╱                                             │
│       │           ╱╱╱╱╱                                                 │
│   90% ┤      ╱╱╱╱╱          ★ BEST                                     │
│       │  ╱╱╱╱       ╲╲╲╲╲╲╲╲ ← Val accuracy (giảm = OVERFITTING)     │
│   85% ┤╱                                                               │
│       │                                                                  │
│   80% ┤                                                                 │
│       └──────────────────────────────                                   │
│       Epoch 0    5    10    15    20                                    │
│                       ↑                                                 │
│              STOP HERE (best val_auc)                                   │
│                                                                         │
│  OVERFITTING = model "học thuộc lòng" training data                    │
│    → accuracy trên train tăng, nhưng trên val GIẢM                     │
│    → model không generalize được sang data mới                          │
│                                                                         │
│  EARLY STOPPING:                                                        │
│    Monitor: val_auc (theo dõi AUC trên validation set)                 │
│    Patience: 5 (chờ 5 epoch, nếu không cải thiện → DỪNG)              │
│    Save best: lưu model tại điểm val_auc cao nhất                     │
│                                                                         │
│  Ví dụ thực tế:                                                         │
│    Epoch 8:  val_auc = 0.91 ★ best → save checkpoint                  │
│    Epoch 9:  val_auc = 0.90 (giảm) → patience counter = 1             │
│    Epoch 10: val_auc = 0.89 → counter = 2                              │
│    Epoch 11: val_auc = 0.92 ★ new best → save, reset counter          │
│    Epoch 12: val_auc = 0.91 → counter = 1                              │
│    ...                                                                   │
│    Epoch 16: val_auc = 0.88 → counter = 5 → STOP!                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: Checkpoint & Resume

### Tại sao cần save/load checkpoint?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CHECKPOINT = "SAVE GAME"                              │
│                                                                         │
│  Vấn đề 1: Colab/Kaggle bị disconnect giữa chừng                      │
│    Train epoch 15/30 → disconnect → MẤT HẾT 6 giờ train               │
│    → CẦN: save checkpoint mỗi epoch → resume từ epoch 15              │
│                                                                         │
│  Vấn đề 2: Muốn dùng model tốt nhất                                    │
│    Epoch 10: val_auc = 0.92 ★ best                                     │
│    Epoch 15: val_auc = 0.88 (overfitting)                              │
│    → CẦN: save best model → dùng model epoch 10 cho inference         │
│                                                                         │
│  Checkpoint lưu GÌ?                                                     │
│  ┌──────────────────────────────────────────────┐                       │
│  │  checkpoint = {                               │                       │
│  │    "epoch": 15,                               │  ← Epoch hiện tại    │
│  │    "model_state_dict": model.state_dict(),    │  ← Weights model     │
│  │    "optimizer_state_dict": optim.state_dict(),│  ← Momentum, LR      │
│  │    "scheduler_state_dict": sched.state_dict(),│  ← LR schedule state │
│  │    "early_stopping_state": es.state_dict(),   │  ← Counter, best     │
│  │    "best_metric": 0.92,                       │  ← Best AUC so far   │
│  │    "scaler_state_dict": scaler.state_dict(),  │  ← AMP scaler state  │
│  │    "config": {...},                            │  ← Toàn bộ config    │
│  │  }                                             │                       │
│  └──────────────────────────────────────────────┘                       │
│                                                                         │
│  2 files:                                                                │
│    outputs/checkpoints/best.pt  ← Model tốt nhất (dùng cho inference) │
│    outputs/checkpoints/last.pt  ← Model cuối cùng (dùng cho resume)   │
│                                                                         │
│  Resume flow:                                                            │
│    1. Kiểm tra last.pt tồn tại?                                        │
│    2. Load tất cả state: model, optimizer, scheduler, early_stopping   │
│    3. Tiếp tục từ epoch tiếp theo                                      │
│    → Quan trọng cho Kaggle/Colab disconnect!                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: Mixed Precision (AMP)

### "Tính nhanh gấp đôi, tốn nửa VRAM" — Tại sao cần?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MIXED PRECISION TRAINING (AMP)                       │
│                                                                         │
│  DEFAULT: float32 (32-bit) — mỗi số chiếm 4 bytes                     │
│    → Chính xác nhưng TỐN VRAM                                         │
│    → RTX 3050 (4GB) → batch_size tối đa ~16                           │
│                                                                         │
│  AMP: Trộn float32 + float16 (16-bit)                                  │
│    → Forward pass: float16 (nhanh gấp 2x trên GPU)                    │
│    → Backward pass: float16 (tiết kiệm VRAM)                          │
│    → Weight update: float32 (giữ precision)                            │
│    → Giảm ~40% VRAM → batch_size lớn hơn → train nhanh hơn           │
│                                                                         │
│  GradScaler: giải quyết vấn đề float16 gradient quá nhỏ               │
│    → Scale loss lên lớn trước backward (tránh underflow)               │
│    → Scale gradient xuống nhỏ lại trước optimizer step                 │
│    → Tự động điều chỉnh scale factor                                   │
│                                                                         │
│  Code pattern:                                                           │
│    scaler = torch.amp.GradScaler(enabled=use_amp)                      │
│    with torch.autocast(device_type="cuda", enabled=use_amp):           │
│        logits = model(images)                                           │
│        loss = loss_fn(logits.squeeze(1), labels)                       │
│    scaler.scale(loss).backward()                                        │
│    scaler.step(optimizer)                                                │
│    scaler.update()                                                       │
│                                                                         │
│  ⚠️ AMP CHỈ hoạt động trên GPU (CUDA). CPU tự động disable.          │
│  ⚠️ Không ảnh hưởng accuracy — chỉ nhanh hơn.                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kiến thức nền: W&B Experiment Tracking

### "Nhật ký phòng thí nghiệm" — Tại sao cần tracking?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WEIGHTS & BIASES (W&B)                                │
│                                                                         │
│  Vấn đề: Train xong 30 epoch, quên LR dùng bao nhiêu?                 │
│  Vấn đề: Train 5 lần khác config, lần nào tốt nhất?                   │
│  Vấn đề: Train trên Kaggle, kết quả ở đâu?                            │
│                                                                         │
│  W&B giải quyết tất cả:                                                 │
│    1. LOG METRICS: loss, accuracy, AUC mỗi epoch → biểu đồ tự động    │
│    2. LOG CONFIG: hyperparameters (LR, batch, epochs) → so sánh runs   │
│    3. DASHBOARD: web UI xem biểu đồ real-time từ bất kỳ đâu           │
│    4. COMPARE: so sánh nhiều runs side-by-side                          │
│                                                                         │
│  Setup (đã làm ở Task 1.1):                                             │
│    wandb login              ← Nhập API key 1 lần                       │
│    wandb.init(project="holmhz")                                         │
│    wandb.log({"loss": 0.5, "auc": 0.85}, step=epoch)                   │
│    wandb.finish()                                                        │
│                                                                         │
│  URL: https://wandb.ai/<username>/holmhz                               │
│                                                                         │
│  ⚠️ W&B là OPTIONAL — nếu không có internet, train vẫn chạy bình thường│
│  ⚠️ Code phải handle: wandb không cài / không login / offline mode     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tổng quan các bước

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TASK 1.5 — ROADMAP                                    │
│                                                                         │
│  Bước 0   Chuẩn bị Git branch                              (~5 phút)   │
│  Bước 1   Implement Metrics (accuracy.py + auc.py)          (~20 phút)  │
│  Bước 2   Implement Loss (losses/bce.py)                    (~15 phút)  │
│  Bước 3   Implement Logger (utils/logger.py)                (~10 phút)  │
│  Bước 4   Implement LR Scheduler (lr_schedulers.py)         (~15 phút)  │
│  Bước 5   Implement Early Stopping (early_stopping.py)      (~25 phút)  │
│  Bước 6   Implement Trainer (trainer.py)                    (~60 phút)  │
│  Bước 7   Update __init__.py exports                        (~10 phút)  │
│  Bước 8   Implement scripts/train.py                        (~30 phút)  │
│  Bước 9   Unit tests                                        (~30 phút)  │
│  Bước 10  Dry run (2 epochs, 100 ảnh)                       (~15 phút)  │
│  Bước 11  Commit & PR                                       (~10 phút)  │
│                                                                         │
│  Tổng ước tính: ~4-5 giờ (chia ra 2-3 ngày)                            │
│                                                                         │
│  Files sẽ tạo/sửa:                                                      │
│    ✏️  src/holmhz/metrics/accuracy.py         (compute_accuracy)        │
│    ✏️  src/holmhz/metrics/auc.py              (compute_auc)             │
│    ✏️  src/holmhz/metrics/__init__.py         (exports)                 │
│    ✏️  src/holmhz/losses/bce.py               (get_loss_fn factory)     │
│    ✏️  src/holmhz/losses/__init__.py          (exports)                 │
│    ✏️  src/holmhz/utils/logger.py             (get_logger)              │
│    ✏️  src/holmhz/training/lr_schedulers.py   (get_scheduler factory)   │
│    ✏️  src/holmhz/training/early_stopping.py  (EarlyStopping class)     │
│    ✏️  src/holmhz/training/trainer.py         (Trainer class)           │
│    ✏️  src/holmhz/training/__init__.py        (exports)                 │
│    ✏️  scripts/train.py                       (CLI entry point)         │
│    ✏️  tests/test_training.py                 (unit tests)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Bước 0: Chuẩn bị Git branch

```bash
cd R:/_Projects/Eurus_Workspace/HolmHz
.venv\Scripts\activate

# Chuyển về main và pull mới nhất
git checkout main
git pull origin main

# Tạo branch mới cho training pipeline
git checkout -b feat/s1/training-pipeline
```

---

## Bước 1: Implement Metrics (accuracy + AUC)

### Tại sao metrics trước?

Trainer cần metrics để:

1. Log metrics mỗi epoch (W&B dashboard)
2. Early stopping monitor `val_auc` (phải tính AUC trước khi compare)
3. Print progress (người dùng biết model đang tốt lên hay xấu đi)

### Code: `src/holmhz/metrics/accuracy.py`

```python
"""
Accuracy metric cho binary classification.

Accuracy = số dự đoán đúng / tổng số mẫu.
Đơn giản nhất, dễ hiểu nhất, nhưng KHÔNG PHẢI metric tốt nhất
khi data imbalanced (nhiều Real hơn Fake hoặc ngược lại).

Ví dụ: 60% Real, 40% Fake
  → Model luôn đoán "Real" → accuracy = 60% (cao nhưng vô nghĩa!)
  → Vì vậy, dùng AUC (metrics/auc.py) làm metric chính.
"""

import torch


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Tính accuracy từ logits và labels.

    Flow:
        logits → sigmoid → probs → (>threshold?) → preds → so sánh labels

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake)
        threshold: ngưỡng phân loại (default 0.5)

    Returns:
        accuracy: float ∈ [0.0, 1.0]

    Example:
        >>> logits = torch.tensor([2.0, -1.0, 0.5])
        >>> labels = torch.tensor([1.0, 0.0, 1.0])
        >>> compute_accuracy(logits, labels)
        0.6667  # 2/3 đúng (2.0→Fake✓, -1.0→Real✓, 0.5→Fake nhưng sigmoid=0.62✓)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= threshold).float()
        correct = (preds == labels).sum().item()
        total = labels.numel()
        return correct / total if total > 0 else 0.0
```

### Code: `src/holmhz/metrics/auc.py`

```python
"""
AUC (Area Under ROC Curve) — metric CHÍNH của HolmHz.

Tại sao AUC mà không phải Accuracy?
→ AUC đo khả năng PHÂN BIỆT giữa 2 class, KHÔNG phụ thuộc threshold
→ AUC = 1.0: phân biệt hoàn hảo (P(Fake|fake) > P(Fake|real) mọi lúc)
→ AUC = 0.5: đoán ngẫu nhiên (tệ như tung đồng xu)
→ AUC < 0.5: model dự đoán ngược (đổi label sẽ tốt hơn!)

KPI dự án:
  - In-domain AUC ≥ 0.90
  - OOD AUC ≥ 0.75

Dùng sklearn.metrics.roc_auc_score — thư viện chuẩn cho ML metrics.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Tính AUC từ logits và labels.

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake)

    Returns:
        auc: float ∈ [0.0, 1.0]

    Edge cases:
        - Nếu chỉ có 1 class trong batch → trả về 0.5 (không tính được)
        - Batch rất nhỏ → AUC có thể không ổn định

    Example:
        >>> logits = torch.tensor([2.0, -1.0, 3.0, -2.0])
        >>> labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        >>> compute_auc(logits, labels)
        1.0  # Phân biệt hoàn hảo
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Edge case: chỉ có 1 class trong batch (toàn Real hoặc toàn Fake)
        # sklearn sẽ raise error → trả về 0.5 (uncertain)
        if len(np.unique(labels_np)) < 2:
            return 0.5

        return float(roc_auc_score(labels_np, probs))
```

### Code: `src/holmhz/metrics/__init__.py`

```python
"""Metrics module — đo lường performance của model."""

from .accuracy import compute_accuracy
from .auc import compute_auc

__all__ = ["compute_accuracy", "compute_auc"]
```

---

## Bước 2: Implement Loss Function

### Code: `src/holmhz/losses/bce.py`

```python
"""
Loss functions cho HolmHz.

Loss function = "thước đo sai lầm" — loss càng cao, model càng sai.
Mục tiêu training: GIẢM loss (optimizer step giảm loss mỗi iteration).

BCEWithLogitsLoss cho binary classification:
  - Binary: 2 class (Real/Fake)
  - CrossEntropy: đo khoảng cách giữa prediction và ground truth
  - WithLogits: nhận raw logits (chưa sigmoid) → numerical stability

Tại sao cần factory function?
→ Config YAML chỉ cần đổi loss.name → code tự tạo loss phù hợp
→ Sau này thêm Focal Loss (cho imbalanced data) dễ dàng
"""

import torch
import torch.nn as nn


def get_loss_fn(
    name: str = "bce_with_logits",
    pos_weight: float | None = None,
) -> nn.Module:
    """Factory tạo loss function theo tên.

    Args:
        name: tên loss function
            - "bce_with_logits": BCEWithLogitsLoss (mặc định, dùng cho HolmHz)
        pos_weight: trọng số cho class positive (Fake)
            Nếu data imbalanced (ví dụ 60% Real, 40% Fake):
            pos_weight = 60/40 = 1.5 → phạt nặng hơn khi miss Fake
            None = cân bằng (mặc định)

    Returns:
        nn.Module — loss function

    Example:
        >>> loss_fn = get_loss_fn("bce_with_logits")
        >>> logits = torch.tensor([0.5, -0.3])
        >>> labels = torch.tensor([1.0, 0.0])
        >>> loss = loss_fn(logits, labels)
    """
    if name == "bce_with_logits":
        weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=weight)

    raise ValueError(
        f"Unknown loss: '{name}'. Available: ['bce_with_logits']"
    )
```

### Code: `src/holmhz/losses/__init__.py`

```python
"""Loss functions module."""

from .bce import get_loss_fn

__all__ = ["get_loss_fn"]
```

---

## Bước 3: Implement Logger

### Code: `src/holmhz/utils/logger.py`

```python
"""
Logging setup cho HolmHz.

Tại sao không dùng print()?
→ Logger có levels (DEBUG, INFO, WARNING, ERROR)
→ Tự thêm timestamp
→ Dễ redirect output vào file
→ Professional code practice
"""

import logging
import sys


def get_logger(
    name: str = "holmhz",
    level: str = "INFO",
) -> logging.Logger:
    """Tạo logger với format sạch.

    Args:
        name: tên logger (thường dùng __name__)
        level: log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        configured logging.Logger

    Example:
        >>> logger = get_logger("training")
        >>> logger.info("Epoch 1: loss=0.5")
        2026-02-26 10:00:00 | INFO     | Epoch 1: loss=0.5
    """
    logger = logging.getLogger(name)

    # Tránh thêm handler trùng nếu gọi nhiều lần
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper()))
    return logger
```

> **Lưu ý**: Đã có `rich` installed, nhưng dùng standard logging đơn giản hơn và không có dependency issue. `rich` sẽ được dùng ở nơi khác (progress bars, tables).

---

## Bước 4: Implement LR Scheduler

### Code: `src/holmhz/training/lr_schedulers.py`

```python
"""
Learning Rate Scheduler factory.

Scheduler điều chỉnh learning rate trong quá trình training.
Cosine Annealing: LR giảm theo đường cong cosine từ lr_max → lr_min.

Tại sao cần factory?
→ Config YAML chỉ đổi training.scheduler: cosine → code tự tạo
→ Sau này thêm StepLR, ReduceLROnPlateau dễ dàng
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def get_scheduler(
    optimizer: Optimizer,
    name: str = "cosine",
    epochs: int = 30,
    eta_min: float = 1e-6,
) -> LRScheduler:
    """Factory tạo LR scheduler theo tên.

    Args:
        optimizer: PyTorch optimizer (đã tạo trước)
        name: tên scheduler
            - "cosine": CosineAnnealingLR (mặc định)
        epochs: tổng số epochs (T_max cho CosineAnnealing)
        eta_min: learning rate tối thiểu cuối cùng

    Returns:
        LR scheduler

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.001)
        >>> scheduler = get_scheduler(optimizer, "cosine", epochs=30)
        >>> # Mỗi epoch:
        >>> scheduler.step()  # LR giảm theo cosine curve
    """
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )

    raise ValueError(
        f"Unknown scheduler: '{name}'. Available: ['cosine']"
    )
```

> **Lưu ý**: `LRScheduler` (không có dấu `_`) là tên mới trong PyTorch 2.x. Nếu import error, thử `from torch.optim.lr_scheduler import _LRScheduler as LRScheduler`.

---

## Bước 5: Implement Early Stopping

### Code: `src/holmhz/training/early_stopping.py`

```python
"""
Early Stopping — dừng training khi metric không cải thiện.

Tại sao cần?
→ Chống overfitting: model "học thuộc" train data nhưng fail trên val
→ Tiết kiệm thời gian: không cần train hết 30 epoch nếu đã hội tụ
→ Tiết kiệm GPU quota: Kaggle 30h/tuần, Colab 4h/session

Pattern từ:
- CNNDetection: earlystop.py (đơn giản, patience-based)
- DeepfakeBench: trainer callback (phức tạp hơn)
- HolmHz: giữ đơn giản + thêm state_dict cho checkpoint resume
"""


class EarlyStopping:
    """Dừng training khi metric không cải thiện sau `patience` epochs.

    Dùng cho monitor=val_auc (mode="max" — AUC càng cao càng tốt).

    Args:
        patience: số epochs chờ trước khi dừng (default 5)
        mode: "max" (metric cao = tốt) hoặc "min" (metric thấp = tốt)
        min_delta: cải thiện tối thiểu để tính là "cải thiện" (default 0.0)

    Example:
        >>> es = EarlyStopping(patience=5, mode="max")
        >>> es(0.85)  # First epoch → always best
        False
        >>> es(0.86)  # Improved → reset counter
        False
        >>> es(0.84)  # Worse → counter = 1
        False
        >>> # ... 4 more epochs without improvement ...
        >>> es(0.83)  # counter = 5 → STOP!
        True
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        # Internal state
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False
        self.is_best = False

    def __call__(self, metric: float) -> bool:
        """Kiểm tra metric mới, cập nhật state.

        Args:
            metric: giá trị metric mới (ví dụ val_auc)

        Returns:
            True nếu nên dừng (patience hết), False nếu tiếp tục
        """
        if self.best_score is None:
            # Epoch đầu tiên — luôn là best
            self.best_score = metric
            self.is_best = True
        elif self._is_improvement(metric):
            # Metric cải thiện → reset counter
            self.best_score = metric
            self.counter = 0
            self.is_best = True
        else:
            # Không cải thiện → tăng counter
            self.counter += 1
            self.is_best = False
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        """Kiểm tra score mới có "đủ tốt hơn" best_score không."""
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:  # mode == "min"
            return score < self.best_score - self.min_delta

    def state_dict(self) -> dict:
        """Lưu state cho checkpoint resume.

        ⚠️ QUAN TRỌNG: Nếu không save state, resume sẽ reset counter
        → model train thêm patience epochs vô nghĩa.
        """
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state từ checkpoint.

        Gọi khi resume training để tiếp tục đếm patience.
        """
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.should_stop = state["should_stop"]

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, mode='{self.mode}', "
            f"counter={self.counter}, best={self.best_score})"
        )
```

---

## Bước 6: Implement Trainer

### Đây là file QUAN TRỌNG NHẤT — orchestrate toàn bộ training

### Code: `src/holmhz/training/trainer.py`

```python
"""
Trainer class — Orchestrate toàn bộ training pipeline.

Trainer là "nhạc trưởng" điều phối:
  DataLoader → Model → Loss → Optimizer → Metrics → Logging → Checkpoint

Pattern từ:
- DeepfakeBench: trainer/trainer.py (base class phức tạp)
- CNNDetection: train.py (script đơn giản, không class)
- HolmHz: Lấy ý tưởng DeepfakeBench nhưng ĐƠN GIẢN HÓA

Flow mỗi epoch:
  1. train_one_epoch() → iterate train_loader, compute loss, backward
  2. validate()        → iterate val_loader, compute metrics (no gradient)
  3. scheduler.step()  → giảm learning rate
  4. early_stopping()  → kiểm tra val_auc có cải thiện không
  5. save_checkpoint() → lưu best.pt và last.pt
  6. wandb.log()       → log metrics lên dashboard

Checkpoint resume:
  - Nếu last.pt tồn tại → tự động resume từ epoch tiếp theo
  - Quan trọng cho Kaggle/Colab bị disconnect giữa chừng
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..metrics.accuracy import compute_accuracy
from ..metrics.auc import compute_auc
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Orchestrate training: train loop, validation, logging, checkpoint.

    Args:
        model: detector model (từ Task 1.4)
        train_loader: training DataLoader (từ Task 1.3)
        val_loader: validation DataLoader (từ Task 1.3)
        optimizer: PyTorch optimizer (AdamW)
        scheduler: LR scheduler (CosineAnnealing)
        loss_fn: loss function (BCEWithLogitsLoss)
        early_stopping: EarlyStopping instance
        config: dict config (từ OmegaConf)
        device: torch.device (cuda hoặc cpu)
        checkpoint_dir: thư mục lưu checkpoint
        use_wandb: bật/tắt W&B logging
        use_amp: bật/tắt Mixed Precision (AMP)

    Example:
        >>> trainer = Trainer(model, train_loader, val_loader, ...)
        >>> trainer.fit(epochs=30)
        # → Train 30 epochs, save best.pt + last.pt
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        early_stopping,
        config: dict,
        device: torch.device,
        checkpoint_dir: str = "outputs/checkpoints",
        use_wandb: bool = True,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.early_stopping = early_stopping
        self.config = config
        self.device = device
        self.use_wandb = use_wandb

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed Precision — chỉ bật trên CUDA
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # State tracking
        self.start_epoch = 0
        self.best_metric = 0.0
        self.history: list[dict] = []

    # ──────────────────────────────────────────────────────────
    # TRAIN ONE EPOCH
    # ──────────────────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> dict:
        """Train model trên toàn bộ training set (1 epoch).

        Flow mỗi batch:
          1. Load images + labels → GPU
          2. Forward pass (AMP autocast)
          3. Compute loss
          4. Backward pass (GradScaler)
          5. Optimizer step
          6. Accumulate metrics

        Args:
            epoch: epoch index (0-based)

        Returns:
            dict với train_loss, train_acc, train_auc
        """
        self.model.train()  # Bật Dropout, BatchNorm training mode
        total_loss = 0.0
        all_logits = []
        all_labels = []
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            leave=False,
        )

        for batch in pbar:
            # Chuyển data lên device (GPU/CPU)
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # ─── Forward pass (Mixed Precision) ───
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                logits = self.model(images)                # [B, 1]
                loss = self.loss_fn(logits.squeeze(1), labels)  # [B]

            # ─── Backward pass (GradScaler) ───
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ─── Accumulate ───
            total_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())
            num_batches += 1

            # Progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Epoch-level metrics
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        return {
            "train_loss": total_loss / max(num_batches, 1),
            "train_acc": compute_accuracy(all_logits, all_labels),
            "train_auc": compute_auc(all_logits, all_labels),
        }

    # ──────────────────────────────────────────────────────────
    # VALIDATE
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Evaluate model trên validation set (không tính gradient).

        @torch.no_grad() = tắt gradient tracking → tiết kiệm VRAM + nhanh hơn.

        Args:
            epoch: epoch index (0-based)

        Returns:
            dict với val_loss, val_acc, val_auc
        """
        self.model.eval()  # Tắt Dropout, dùng running stats BatchNorm
        total_loss = 0.0
        all_logits = []
        all_labels = []
        num_batches = 0

        for batch in tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1} [Val]",
            leave=False,
        ):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            logits = self.model(images)                      # [B, 1]
            loss = self.loss_fn(logits.squeeze(1), labels)   # scalar

            total_loss += loss.item()
            all_logits.append(logits)
            all_labels.append(labels)
            num_batches += 1

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_acc": compute_accuracy(all_logits, all_labels),
            "val_auc": compute_auc(all_logits, all_labels),
        }

    # ──────────────────────────────────────────────────────────
    # FIT (main training loop)
    # ──────────────────────────────────────────────────────────

    def fit(self, epochs: int) -> list[dict]:
        """Main training loop — chạy train + validate cho mỗi epoch.

        Args:
            epochs: tổng số epochs (thường 30)

        Returns:
            list[dict] — history metrics cho từng epoch
        """
        logger.info("=" * 60)
        logger.info("TRAINING START")
        logger.info("=" * 60)
        logger.info(f"Epochs: {self.start_epoch + 1} → {epochs}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples:   {len(self.val_loader.dataset)}")
        logger.info(f"Device: {self.device}")
        logger.info(f"AMP: {self.use_amp}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")
        logger.info("=" * 60)

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            # ─── Train ───
            train_metrics = self.train_one_epoch(epoch)

            # ─── Validate ───
            val_metrics = self.validate(epoch)

            # ─── Scheduler step ───
            self.scheduler.step()

            # ─── Combine metrics ───
            lr = self.optimizer.param_groups[0]["lr"]
            metrics = {
                **train_metrics,
                **val_metrics,
                "lr": lr,
                "epoch": epoch,
                "epoch_time": time.time() - epoch_start,
            }
            self.history.append(metrics)

            # ─── Log to console ───
            self._log_epoch(metrics, epoch)

            # ─── Log to W&B ───
            if self.use_wandb:
                self._log_wandb(metrics, epoch)

            # ─── Early stopping ───
            monitor_key = "val_auc"
            monitor_value = val_metrics.get(monitor_key, 0.0)
            self.early_stopping(monitor_value)

            # ─── Save checkpoints ───
            if self.early_stopping.is_best:
                self.best_metric = monitor_value
                self.save_checkpoint(epoch, is_best=True)

            # Always save last (cho resume)
            self.save_checkpoint(epoch, is_best=False)

            if self.early_stopping.should_stop:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {self.early_stopping.patience} epochs)"
                )
                break

        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE — Best val_auc: {self.best_metric:.4f}")
        logger.info("=" * 60)

        return self.history

    # ──────────────────────────────────────────────────────────
    # LOGGING
    # ──────────────────────────────────────────────────────────

    def _log_epoch(self, metrics: dict, epoch: int) -> None:
        """Print metrics đẹp ra console."""
        best_marker = " ★" if self.early_stopping.is_best else ""
        logger.info(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Val Acc: {metrics['val_acc']:.4f} | "
            f"Val AUC: {metrics['val_auc']:.4f}{best_marker} | "
            f"LR: {metrics['lr']:.2e} | "
            f"{metrics['epoch_time']:.1f}s"
        )

    def _log_wandb(self, metrics: dict, epoch: int) -> None:
        """Log metrics lên W&B dashboard."""
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=epoch)
        except ImportError:
            pass

    # ──────────────────────────────────────────────────────────
    # CHECKPOINT SAVE / LOAD
    # ──────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Lưu checkpoint (model + optimizer + scheduler + state).

        Args:
            epoch: epoch hiện tại
            is_best: True → save thêm best.pt
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "early_stopping_state_dict": self.early_stopping.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
            "scaler_state_dict": (
                self.scaler.state_dict() if self.use_amp else None
            ),
        }

        # Always save last (cho resume)
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(state, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            logger.info(
                f"  ★ New best model saved (val_auc={self.best_metric:.4f})"
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint và resume training.

        Khôi phục TOÀN BỘ state: model, optimizer, scheduler, early_stopping.
        Training sẽ tiếp tục từ epoch tiếp theo.

        Args:
            checkpoint_path: đường dẫn tới file .pt
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "early_stopping_state_dict" in checkpoint:
            self.early_stopping.load_state_dict(
                checkpoint["early_stopping_state_dict"]
            )

        self.best_metric = checkpoint.get("best_metric", 0.0)
        self.start_epoch = checkpoint["epoch"] + 1

        if self.use_amp and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Resumed from epoch {self.start_epoch}, "
            f"best_metric={self.best_metric:.4f}"
        )
```

---

## Bước 7: Update **init**.py exports

### Code: `src/holmhz/training/__init__.py`

```python
"""Training module — Trainer, Early Stopping, LR Schedulers."""

from .early_stopping import EarlyStopping
from .lr_schedulers import get_scheduler
from .trainer import Trainer

__all__ = ["Trainer", "EarlyStopping", "get_scheduler"]
```

> **Cũng cần update** `src/holmhz/utils/__init__.py` nếu chưa export logger:

```python
# src/holmhz/utils/__init__.py
"""Utility modules — Registry, logging, I/O helpers."""

from .logger import get_logger
from .registry import BACKBONE_REGISTRY, DETECTOR_REGISTRY, Registry

__all__ = [
    "Registry",
    "BACKBONE_REGISTRY",
    "DETECTOR_REGISTRY",
    "get_logger",
]
```

---

## Bước 8: Implement scripts/train.py

### Code: `scripts/train.py`

```python
"""
HolmHz Training Script — CLI entry point.

Usage:
    # Default config (configs/train.yaml)
    python scripts/train.py

    # Custom config
    python scripts/train.py configs/train.yaml

    # Override specific values
    python scripts/train.py --batch_size 16 --epochs 5

Example dry run (local, 2 epochs):
    python scripts/train.py --epochs 2 --num_workers 0 --batch_size 8

Full training (Kaggle GPU):
    python scripts/train.py  # Uses default config
"""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# ─── Import HolmHz modules ───
import holmhz.detectors  # Trigger DETECTOR_REGISTRY registration  # noqa: F401
from holmhz.data import create_dataloader
from holmhz.losses import get_loss_fn
from holmhz.training import EarlyStopping, Trainer, get_scheduler
from holmhz.utils.logger import get_logger
from holmhz.utils.registry import DETECTOR_REGISTRY

logger = get_logger("train")


def main():
    """Main training entry point."""
    # ─── Load config ───
    config_path = "configs/train.yaml"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        config_path = sys.argv[1]

    config = OmegaConf.load(config_path)

    # CLI overrides (--key value)
    cli_overrides = OmegaConf.from_cli(sys.argv[1:])
    config = OmegaConf.merge(config, cli_overrides)

    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # ─── Device ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ─── Data ───
    train_loader = create_dataloader(
        manifest_path=config.data.train_manifest,
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        is_training=True,
        num_workers=config.data.num_workers,
    )
    val_loader = create_dataloader(
        manifest_path=config.data.val_manifest,
        batch_size=config.training.batch_size * 2,  # Val batch lớn hơn (no gradient)
        image_size=config.data.image_size,
        is_training=False,
        num_workers=config.data.num_workers,
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # ─── Model ───
    model = DETECTOR_REGISTRY.build(
        config.model.name,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    # ─── Optimizer ───
    # Chỉ optimize params có requires_grad (head khi freeze backbone)
    params = [p for p in model.parameters() if p.requires_grad]

    if config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=config.training.learning_rate,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    # ─── Scheduler ───
    scheduler = get_scheduler(
        optimizer,
        name=config.training.scheduler,
        epochs=config.training.epochs,
    )

    # ─── Loss ───
    loss_fn = get_loss_fn("bce_with_logits")

    # ─── Early Stopping ───
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping.patience,
        mode="max",  # val_auc — higher is better
    )

    # ─── W&B ───
    use_wandb = False
    try:
        import wandb

        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.get("entity"),
            config=OmegaConf.to_container(config, resolve=True),
        )
        use_wandb = True
        logger.info(f"W&B run: {wandb.run.name}")
    except Exception as e:
        logger.warning(f"W&B disabled: {e}")

    # ─── Trainer ───
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        early_stopping=early_stopping,
        config=OmegaConf.to_container(config, resolve=True),
        device=device,
        use_wandb=use_wandb,
    )

    # ─── Resume from checkpoint ───
    resume_path = Path("outputs/checkpoints/last.pt")
    if resume_path.exists():
        trainer.load_checkpoint(str(resume_path))

    # ─── Train ───
    trainer.fit(config.training.epochs)

    # ─── Cleanup ───
    if use_wandb:
        import wandb

        wandb.finish()

    logger.info("Done!")


if __name__ == "__main__":
    main()
```

---

## Bước 9: Unit tests

### Code: `tests/test_training.py`

```python
"""Unit tests cho training pipeline.

Kiểm tra:
- Metrics (accuracy, AUC) tính đúng
- Loss function factory hoạt động
- EarlyStopping logic đúng (patience, state_dict)
- LR Scheduler factory hoạt động
- Trainer forward/backward pass (1 batch)
"""

import pytest
import torch
import torch.nn as nn

from holmhz.metrics import compute_accuracy, compute_auc
from holmhz.losses import get_loss_fn
from holmhz.training import EarlyStopping, get_scheduler


class TestMetrics:
    """Test accuracy và AUC computation."""

    def test_accuracy_perfect(self):
        """Model dự đoán hoàn hảo → accuracy = 1.0."""
        logits = torch.tensor([5.0, -5.0, 5.0, -5.0])  # Rõ ràng Fake, Real
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_accuracy(logits, labels) == 1.0

    def test_accuracy_random(self):
        """Model dự đoán ngược → accuracy = 0.0."""
        logits = torch.tensor([-5.0, 5.0])  # Ngược hết
        labels = torch.tensor([1.0, 0.0])
        assert compute_accuracy(logits, labels) == 0.0

    def test_accuracy_with_2d_logits(self):
        """Logits shape [B, 1] (từ model) cũng phải hoạt động."""
        logits = torch.tensor([[5.0], [-5.0]])
        labels = torch.tensor([1.0, 0.0])
        assert compute_accuracy(logits, labels) == 1.0

    def test_auc_perfect(self):
        """Phân biệt hoàn hảo → AUC = 1.0."""
        logits = torch.tensor([5.0, -5.0, 5.0, -5.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_auc(logits, labels) == 1.0

    def test_auc_random(self):
        """Chỉ 1 class → AUC = 0.5 (edge case)."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        labels = torch.tensor([1.0, 1.0, 1.0])  # Toàn Fake
        assert compute_auc(logits, labels) == 0.5

    def test_auc_with_2d_logits(self):
        """Logits shape [B, 1] cũng phải hoạt động."""
        logits = torch.tensor([[5.0], [-5.0], [3.0], [-3.0]])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_auc(logits, labels) == 1.0


class TestLossFunction:
    """Test loss function factory."""

    def test_bce_with_logits(self):
        """BCEWithLogitsLoss phải tạo được và tính loss."""
        loss_fn = get_loss_fn("bce_with_logits")
        logits = torch.tensor([0.0, 0.0])  # Uncertain
        labels = torch.tensor([1.0, 0.0])
        loss = loss_fn(logits, labels)
        # loss ≈ 0.693 (log(2)) khi logits = 0
        assert 0.6 < loss.item() < 0.8

    def test_bce_with_pos_weight(self):
        """pos_weight parameter phải hoạt động."""
        loss_fn = get_loss_fn("bce_with_logits", pos_weight=2.0)
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_unknown_loss_raises(self):
        """Loss không tồn tại phải raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss_fn("unknown_loss")


class TestEarlyStopping:
    """Test Early Stopping logic."""

    def test_first_epoch_is_best(self):
        """Epoch đầu luôn là best."""
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)
        assert es.is_best
        assert not es.should_stop

    def test_improvement_resets_counter(self):
        """Metric cải thiện → reset counter."""
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)
        es(0.4)  # Worse → counter=1
        assert es.counter == 1
        es(0.6)  # Better → counter=0
        assert es.counter == 0
        assert es.is_best

    def test_patience_triggers_stop(self):
        """Hết patience → should_stop = True."""
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)  # Best
        es(0.4)  # counter=1
        es(0.3)  # counter=2
        result = es(0.2)  # counter=3 → STOP
        assert result is True
        assert es.should_stop

    def test_mode_min(self):
        """mode='min': metric giảm = tốt."""
        es = EarlyStopping(patience=3, mode="min")
        es(0.5)  # Best
        es(0.3)  # Better (lower)
        assert es.is_best
        assert es.best_score == 0.3

    def test_state_dict_roundtrip(self):
        """state_dict save/load phải giữ nguyên state."""
        es = EarlyStopping(patience=5, mode="max")
        es(0.5)
        es(0.6)
        es(0.55)  # Worse → counter=1

        state = es.state_dict()
        assert state["counter"] == 1
        assert state["best_score"] == 0.6

        es2 = EarlyStopping(patience=5, mode="max")
        es2.load_state_dict(state)
        assert es2.counter == 1
        assert es2.best_score == 0.6


class TestLRScheduler:
    """Test LR Scheduler factory."""

    def test_cosine_scheduler(self):
        """CosineAnnealingLR phải tạo được."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = get_scheduler(optimizer, "cosine", epochs=10)

        # LR phải giảm sau step
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]
        assert lr_after < lr_before

    def test_unknown_scheduler_raises(self):
        """Scheduler không tồn tại phải raise ValueError."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler(optimizer, "unknown_scheduler")
```

> **Chạy tests:**
>
> ```bash
> pytest tests/test_training.py -v
> ```

---

## Bước 10: Dry run (2 epochs, 100 ảnh)

Sau khi tất cả tests pass, chạy thử training thực sự với data nhỏ:

```bash
# Dry run: 2 epochs, batch=8, num_workers=0 (Windows safe)
python scripts/train.py --training.epochs 2 --training.batch_size 8 --data.num_workers 0
```

**Expected output:**

```
2026-02-XX HH:MM:SS | INFO     | Config loaded from: configs/train.yaml
2026-02-XX HH:MM:SS | INFO     | Device: cuda
2026-02-XX HH:MM:SS | INFO     | Train: 18550 samples, 579 batches
2026-02-XX HH:MM:SS | INFO     | Val:   3975 samples, 249 batches
2026-02-XX HH:MM:SS | INFO     | Model: efficientnet_b0
2026-02-XX HH:MM:SS | INFO     | Total params: 4,008,829
2026-02-XX HH:MM:SS | INFO     | Trainable params: 1,281
...
2026-02-XX HH:MM:SS | INFO     | Epoch   1 | Train Loss: 0.69xx | Val Loss: 0.69xx | Val Acc: 0.5xxx | Val AUC: 0.5xxx ★ | LR: 1.00e-03 | XXs
2026-02-XX HH:MM:SS | INFO     | Epoch   2 | Train Loss: 0.68xx | Val Loss: 0.68xx | Val Acc: 0.5xxx | Val AUC: 0.5xxx   | LR: 9.99e-04 | XXs
...
2026-02-XX HH:MM:SS | INFO     | TRAINING COMPLETE — Best val_auc: 0.5xxx
```

> **Lưu ý cho dry run local (RTX 3050, 4GB VRAM)**:
>
> - Dùng `batch_size=8` thay vì 32 (tránh OOM)
> - Dùng `num_workers=0` (Windows MINGW64 multiprocessing issue)
> - `pretrained=True` lần đầu sẽ download weights (~20MB)
> - AUC ~0.5 là bình thường (model chưa train đủ, chỉ 2 epoch)

**Kiểm tra checkpoint:**

```bash
ls outputs/checkpoints/
# best.pt   last.pt
```

**Test resume:**

```bash
# Chạy lại → phải resume từ epoch 3
python scripts/train.py --training.epochs 4 --training.batch_size 8 --data.num_workers 0
# Expected: "Resumed from epoch 2, best_metric=0.5xxx"
```

---

## Bước 11: Commit & PR

```bash
# 1. Lint check
ruff check src/holmhz/metrics/ src/holmhz/losses/ src/holmhz/training/ src/holmhz/utils/
ruff check scripts/train.py tests/test_training.py

# 2. Run ALL tests (data + model + training)
pytest tests/ -v

# 3. Stage files
git add src/holmhz/metrics/ src/holmhz/losses/ src/holmhz/training/ src/holmhz/utils/
git add scripts/train.py tests/test_training.py

# 4. Commit
git commit -m "feat(training): implement training pipeline with Trainer, metrics, checkpoint resume

- compute_accuracy() + compute_auc() metrics
- BCEWithLogitsLoss factory (losses/bce.py)
- CosineAnnealingLR scheduler factory
- EarlyStopping with state_dict (patience=5, monitor=val_auc)
- Trainer class: train/val loop, AMP, W&B logging, checkpoint save/resume
- scripts/train.py CLI entry point (OmegaConf config)
- Logger setup (utils/logger.py)
- Unit tests: test_training.py"

# 5. Push
git push origin feat/s1/training-pipeline
```

---

## Checklist hoàn thành

Trước khi đánh dấu Task 1.5 ✅ DONE:

### Code implementation

- [ ] `src/holmhz/metrics/accuracy.py` — `compute_accuracy(logits, labels)`
- [ ] `src/holmhz/metrics/auc.py` — `compute_auc(logits, labels)`
- [ ] `src/holmhz/metrics/__init__.py` — exports
- [ ] `src/holmhz/losses/bce.py` — `get_loss_fn()` factory
- [ ] `src/holmhz/losses/__init__.py` — exports
- [ ] `src/holmhz/utils/logger.py` — `get_logger()` setup
- [ ] `src/holmhz/training/lr_schedulers.py` — `get_scheduler()` factory
- [ ] `src/holmhz/training/early_stopping.py` — `EarlyStopping` class
- [ ] `src/holmhz/training/trainer.py` — `Trainer` class
- [ ] `src/holmhz/training/__init__.py` — exports
- [ ] `scripts/train.py` — CLI entry point

### Functionality

- [ ] BCEWithLogitsLoss nhận logits `[B, 1]` squeeze → `[B]`, labels `[B]`
- [ ] Metrics: accuracy + AUC tính đúng (verified qua unit tests)
- [ ] CosineAnnealingLR: LR giảm từ 0.001 → 1e-6 theo cosine
- [ ] Early stopping: patience=5, monitor=val_auc, mode="max"
- [ ] Checkpoint save: `outputs/checkpoints/best.pt` + `last.pt`
- [ ] Checkpoint resume: load all state, tiếp tục training từ epoch tiếp
- [ ] Mixed precision (AMP): GradScaler + autocast, auto-disable trên CPU
- [ ] W&B logging: optional, handle ImportError gracefully

### Tests

- [ ] `pytest tests/test_training.py -v` → tất cả PASSED
- [ ] `ruff check src/ tests/` → clean (hoặc chỉ warnings nhẹ)

### Dry run

- [ ] `python scripts/train.py --training.epochs 2 --training.batch_size 8 --data.num_workers 0` → chạy 2 epoch, save checkpoint
- [ ] Resume: chạy lại → tiếp tục từ epoch 3
- [ ] `outputs/checkpoints/best.pt` + `last.pt` tồn tại

### Git

- [ ] Branch: `feat/s1/training-pipeline`
- [ ] Code committed
- [ ] PR Created trên GitHub

---

## Troubleshooting

### Q: `BrokenPipeError` hoặc `RuntimeError: DataLoader worker exited unexpectedly` trên Windows

**A**: Windows + MINGW64 + `num_workers > 0` thường bị lỗi. Fix:

```bash
# Dùng num_workers=0 trên local Windows
python scripts/train.py --data.num_workers 0
```

Trên Kaggle/Colab (Linux): `num_workers=4` hoạt động bình thường.

### Q: `CUDA out of memory` (OOM)

**A**: RTX 3050 chỉ có 4GB VRAM. Giảm batch size:

```bash
python scripts/train.py --training.batch_size 8
```

Nếu vẫn OOM:

- `batch_size=4`
- Kiểm tra AMP đang bật: code đã tự bật khi device=cuda
- Tắt `pin_memory` trong `create_dataloader` (hiếm khi cần)

### Q: `ImportError: No module named 'wandb'`

**A**: W&B là optional. Code đã handle gracefully:

```python
try:
    import wandb
    wandb.init(...)
except Exception:
    use_wandb = False  # Tự tắt, training vẫn chạy
```

Nếu muốn cài: `pip install wandb && wandb login`

### Q: Loss không giảm sau nhiều epoch

**A**: Kiểm tra:

1. **LR quá cao/thấp**: Thử `--training.learning_rate 0.0001` hoặc `0.01`
2. **Data augmentation quá mạnh**: Tạm tắt: `--data.augmentation false`
3. **Model frozen**: Nếu `freeze_backbone=true`, chỉ 1,281 params train → loss giảm chậm là bình thường
4. **Data bị hỏng**: Chạy lại `scripts/validate_dataset.py`

### Q: `torch.autocast` hoặc `torch.amp.GradScaler` không tìm thấy

**A**: Cần PyTorch ≥ 2.0. Kiểm tra:

```python
import torch; print(torch.__version__)  # Phải ≥ 2.0
```

Nếu PyTorch cũ hơn, dùng API cũ:

```python
from torch.cuda.amp import GradScaler, autocast
```

### Q: `OmegaConf.from_cli()` không nhận args

**A**: Format CLI override phải là `key=value` (không có `--`):

```bash
# ✅ Đúng (OmegaConf format)
python scripts/train.py training.epochs=5 training.batch_size=8

# ❌ Sai (argparse format — không dùng)
python scripts/train.py --epochs 5 --batch_size 8
```

Thực ra code đã handle cả 2 format — `OmegaConf.from_cli()` sẽ parse `--training.epochs 2` thành `training.epochs: 2`.

### Q: Checkpoint resume không hoạt động

**A**: Kiểm tra:

1. File `outputs/checkpoints/last.pt` tồn tại?
2. Config có thay đổi model architecture? (Nếu đổi model → state_dict incompatible)
3. Lỗi cụ thể? `weights_only=False` đã set để load pickle

### Q: `LRScheduler` import error

**A**: Trong PyTorch 2.x, tên class thay đổi:

```python
# PyTorch 2.4+
from torch.optim.lr_scheduler import LRScheduler

# PyTorch 2.0-2.3 (fallback)
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
```

---

## Mối liên hệ với các Task tiếp theo

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SAU TASK 1.5 — CÁC BƯỚC TIẾP THEO                   │
│                                                                         │
│  Task 1.3 (✅) tạo ra:                                                 │
│  • DataLoader: train (18,550), val (3,975), test_id (3,975), OOD (1,180)│
│                                                                         │
│  Task 1.4 (✅) tạo ra:                                                 │
│  • EfficientNetDetector (4M params)                                     │
│  • Registry pattern                                                     │
│                                                                         │
│  Task 1.5 (✅) tạo ra:                                                 │
│  • Trainer class                                                        │
│  • scripts/train.py                                                     │
│  • Metrics, Loss, Scheduler, EarlyStopping                              │
│  • Checkpoint save/resume                                               │
│                                                                         │
│  Task 1.6 SẼ LÀM:                                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Phase 1: Freeze backbone + train head                            │  │
│  │    python scripts/train.py                                        │  │
│  │    # freeze_backbone=true, lr=0.001, epochs=30                    │  │
│  │    # Kaggle GPU: T4, batch=32, ~15 min/epoch                     │  │
│  │    # Target: val_auc ≥ 0.85                                      │  │
│  │                                                                    │  │
│  │  Phase 2: Unfreeze backbone + fine-tune all                       │  │
│  │    python scripts/train.py --model.freeze_backbone false \        │  │
│  │                            --training.learning_rate 0.0001        │  │
│  │    # LR thấp hơn 10x cho backbone (tránh catastrophic forgetting)│  │
│  │    # Target: val_auc ≥ 0.90, OOD_auc ≥ 0.75                    │  │
│  │                                                                    │  │
│  │  Save best checkpoint → dùng cho Sprint 2                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Sprint 2:                                                               │
│  • Task 2.1: Eval pipeline (load best.pt, compute metrics trên test)   │
│  • Task 2.2: Benchmark vs 3 SOTA (chạy cùng test set)                 │
│  • Task 2.3: Grad-CAM XAI (dùng model.get_feature_layer())            │
│  • Task 2.4: Export ONNX (dùng best.pt)                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**Last Updated**: 26/02/2026  
**Author**: Generated by GitHub Copilot for Lê Văn Hoàng  
**Version**: 1.0 (aligned with Task 1.3 + 1.4 completed, configs/train.yaml verified)
