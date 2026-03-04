## 💡 Context

> **Task ID**: S1-005  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: ~~07/03/2026~~ → **14/03/2026**  
> **Assignee**: Hoàng  
> **Blocked by**: S1-003 (data pipeline), S1-004 (model)  
> **Blocks**: S1-006 (Baseline training cần pipeline)

> Xây dựng training pipeline: Trainer class, loss, scheduler, early stopping, logging.
> Pattern từ DeepfakeBench trainer/ và CNNDetection earlystop.py.

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Engineer**, I want to **build a complete training pipeline with loss functions, LR scheduling, early stopping, and W&B logging** so that **I can train, monitor, and reproduce experiments efficiently on Kaggle/Colab.**

**Acceptance Criteria:**

- [ ] `Trainer` class orchestrate full train loop (train/val per epoch)
- [ ] BCEWithLogitsLoss hoạt động (nhận logits `[B,1]` từ Task 1.4, squeeze → `[B]` cho labels)
- [ ] Metrics: `compute_accuracy()` + `compute_auc()` (cần cho early stopping monitor=val_auc)
- [ ] CosineAnnealingLR scheduler configured
- [ ] Early stopping: patience=5, monitor=val_auc, hỗ trợ state_dict (cho resume)
- [ ] W&B logging: loss, AUC, accuracy, LR per epoch
- [ ] Checkpoint save: best.pt + last.pt
- [ ] **Checkpoint resume**: load model + optimizer + scheduler + epoch + early_stopping state
- [ ] Mixed precision (AMP): GradScaler + autocast cho GPU training (quan trọng với 4GB VRAM)
- [ ] Config-driven: tất cả hyperparams đọc từ YAML (`configs/train.yaml`)
- [ ] Script `scripts/train.py` chạy được end-to-end (dry run 2 epochs trên 100 ảnh)

---

## 🛠️ Implementation

### Subtasks

- [ ] 1.5.1 Implement `src/holmhz/metrics/accuracy.py` + `auc.py` (compute_accuracy, compute_auc)
- [ ] 1.5.2 Implement `src/holmhz/losses/bce.py` (BCEWithLogitsLoss factory)
- [ ] 1.5.3 Implement `src/holmhz/utils/logger.py` (logging setup)
- [ ] 1.5.4 Implement `src/holmhz/training/lr_schedulers.py` (CosineAnnealing factory)
- [ ] 1.5.5 Implement `src/holmhz/training/early_stopping.py` (patience-based, state_dict support)
- [ ] 1.5.6 Implement `src/holmhz/training/trainer.py` (Trainer class: train/val/fit, W&B, AMP, checkpoint save/resume)
- [ ] 1.5.7 Update `__init__.py` exports (metrics, losses, training, utils)
- [ ] 1.5.8 Implement `scripts/train.py` CLI entry point (OmegaConf + YAML config)
- [ ] 1.5.9 Unit tests + dry run (2 epochs trên 100 ảnh)

### Branch & PR

- [ ] Branch: `feat/s1/training-pipeline`
- [ ] PR Created
- [ ] Dry run on 100 images passed
- [ ] W&B dashboard showing metrics

---

## 📝 Notes

> **Config YAML hiện tại (`configs/train.yaml`) — đã có sẵn từ Task 1.1:**
>
> ```yaml
> model:
>   name: efficientnet_b0
>   pretrained: true
>   num_classes: 1
>   dropout: 0.3
>   freeze_backbone: true
> training:
>   epochs: 30
>   batch_size: 32
>   learning_rate: 0.001
>   optimizer: adamw
>   weight_decay: 0.0001
>   scheduler: cosine
>   early_stopping:
>     patience: 5
>     monitor: val_auc
> data:
>   train_manifest: data/manifests/train.json
>   val_manifest: data/manifests/val.json
>   image_size: 224
>   num_workers: 4
>   augmentation: true
> wandb:
>   project: holmhz
>   entity: null
>   log_every_n_steps: 10
> ```

> **Trainer pattern (từ DeepfakeBench trainer/trainer.py):**
>
> - Base trainer: setup optimizer, scheduler, device
> - Train one epoch → validate → log → early stop check
> - Save checkpoint mỗi khi val metric cải thiện
> - Support resume from checkpoint

> **Lưu ý cho Kaggle/Colab (Cập nhật 24/02/2026):**
>
> - **Kaggle (PRIMARY)**: T4×2 hoặc P100, 30h/tuần, KHÔNG disconnect
>   - `kaggle kernels push` hoặc dùng web UI
>   - Output save vào `/kaggle/working/` → download sau
> - **Colab (BACKUP)**: T4, ~4h/session, CÓ THỂ disconnect
>   - Mount Google Drive cho checkpoint persistence
>   - LUÔN save checkpoint mỗi epoch (quan trọng!)
> - **Local RTX 3050**: 4GB VRAM → batch_size=8-16, dùng để dev & debug only
> - `wandb login` 1 lần, token lưu trong env
