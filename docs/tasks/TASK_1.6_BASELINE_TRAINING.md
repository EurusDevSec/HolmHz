## 💡 Context

> **Task ID**: S1-006  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: 15/03/2026  
> **Assignee**: Hoàng  
> **Blocked by**: S1-005 (training pipeline)  
> **Blocks**: S2-001 (Evaluation cần trained model)  
> **Milestone**: ✅ M1 — Dataset v1 (≥25k) + Baseline AUC ≥ 0.88 (in-domain)

> Train EfficientNet-B0 baseline trên dataset GAN + Diffusion.
> Transfer learning: freeze backbone → train head → unfreeze → fine-tune.

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **train the EfficientNet-B0 baseline model on a mixed GAN + Diffusion dataset** so that **I get a working detector to benchmark against the 3 SOTA models and establish HolmHz's performance floor.**

**Acceptance Criteria:**

- [ ] Phase 1: Freeze backbone, train head only → AUC ≥ 0.80 on val set
- [ ] Phase 2: Unfreeze, fine-tune full model → AUC ≥ 0.88 on val set
- [ ] Hyperparameter search: ≥3 LR values, ≥2 batch sizes tested
- [ ] Best checkpoint saved (`.pt` file)
- [ ] Training curve (loss, AUC) logged on W&B — no overfitting (val loss ≤ 1.2x train loss)
- [ ] Quick smoke test on 5 ảnh từ `imgs/` folder (Gemini, Flux, Real)
- [ ] Training time documented (epochs, wall clock on Colab T4)

---

## 🛠️ Implementation

### Subtasks

- [ ] 1.6.1 Train freeze backbone + head only (LR=1e-3, epochs=10)
- [ ] 1.6.2 Unfreeze + fine-tune full model (LR=1e-4, epochs=20)
- [ ] 1.6.3 Hyperparameter tuning (LR: {5e-4, 1e-4, 5e-5}, batch: {16, 32})
- [ ] 1.6.4 Save best checkpoint + log final metrics

### Branch & PR

- [ ] Branch: `feat/s1/baseline-training`
- [ ] PR Created
- [ ] W&B experiment link attached
- [ ] Best AUC documented

---

## 📝 Notes

> **Training strategy (từ bài học benchmark):**
>
> 1. **Freeze + head**: Nhanh (5 phút/epoch), kiểm tra pipeline chạy đúng
> 2. **Unfreeze + fine-tune**: Chậm hơn (20 phút/epoch), model thực sự học features
> 3. Nếu AUC < 0.85 sau fine-tune → xem lại data balance & augmentation

> **Hyperparameter grid:**
>
> ```
> LR:         [5e-4, 1e-4, 5e-5]
> Batch:      [16, 32]
> Scheduler:  CosineAnnealing (T_max = total_epochs)
> Optimizer:  AdamW (weight_decay=1e-4)
> ```

> **Ước tính thời gian Colab T4:**
>
> - 25k images, batch=32 → ~780 steps/epoch
> - Freeze: ~3 min/epoch × 10 = 30 min
> - Fine-tune: ~15 min/epoch × 20 = 5 hours
> - Tổng: ~6 hours (1 Colab session)

> **Exit criteria cho Milestone 1:**
>
> - AUC ≥ 0.88 in-domain → ✅ Proceed to Sprint 2
> - AUC 0.80-0.88 → Thêm data hoặc augmentation, retry
> - AUC < 0.80 → Xem lại pipeline, có thể bug
