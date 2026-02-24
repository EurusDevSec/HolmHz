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
- [ ] BCEWithLogitsLoss hoạt động (nhận logits, không cần sigmoid trong loss)
- [ ] CosineAnnealingLR scheduler configured
- [ ] Early stopping: patience=5, monitor=val_auc
- [ ] W&B logging: loss, AUC, accuracy, LR per epoch
- [ ] Checkpoint save: best model + last model
- [ ] **Checkpoint resume**: load từ checkpoint + tiếp tục train (quan trọng cho Colab disconnect)
- [ ] Config-driven: tất cả hyperparams đọc từ YAML
- [ ] Script `scripts/train.py` chạy được end-to-end trên small dataset (100 ảnh)

---

## 🛠️ Implementation

### Subtasks

- [ ] 1.5.1 Implement `src/holmhz/training/trainer.py` (train loop, val loop, epoch logic)
- [ ] 1.5.2 Implement `src/holmhz/losses/bce.py` (BCEWithLogitsLoss wrapper)
- [ ] 1.5.3 Implement `src/holmhz/training/lr_schedulers.py` (CosineAnnealing)
- [ ] 1.5.4 Implement `src/holmhz/training/early_stopping.py` (patience-based)
- [ ] 1.5.5 W&B integration trong Trainer (log metrics, save config)
- [ ] 1.5.6 **Checkpoint resume** — load optimizer + scheduler + epoch + best metric
- [ ] 1.5.7 Implement `scripts/train.py` CLI entry point (argparse + YAML config)

### Branch & PR

- [ ] Branch: `feat/s1/training-pipeline`
- [ ] PR Created
- [ ] Dry run on 100 images passed
- [ ] W&B dashboard showing metrics

---

## 📝 Notes

> **Config YAML mẫu (`configs/train.yaml`):**
>
> ```yaml
> model:
>   name: efficientnet_b0
>   pretrained: true
>   freeze_backbone: true
> training:
>   epochs: 30
>   batch_size: 32
>   lr: 1e-3
>   optimizer: adam
>   scheduler: cosine
>   early_stopping:
>     patience: 5
>     monitor: val_auc
> data:
>   train_manifest: data/manifests/train.json
>   val_manifest: data/manifests/val.json
>   image_size: 224
>   num_workers: 4
> wandb:
>   project: holmhz
>   entity: null
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
