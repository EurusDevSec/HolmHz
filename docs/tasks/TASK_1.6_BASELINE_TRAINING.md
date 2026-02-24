## 💡 Context

> **Task ID**: S1-006  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: ~~15/03/2026~~ → **21/03/2026**  
> **Assignee**: Hoàng  
> **Blocked by**: S1-005 (training pipeline)  
> **Blocks**: S2-001 (Evaluation cần trained model)  
> **Milestone**: ✅ M1 — Dataset v1 (≥15k) + Baseline AUC ≥ 0.85 (in-domain)

> Train EfficientNet-B0 baseline trên dataset GAN + Diffusion.
> Transfer learning: freeze backbone → train head → unfreeze → fine-tune.

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **train the EfficientNet-B0 baseline model on a mixed GAN + Diffusion dataset** so that **I get a working detector to benchmark against the 3 SOTA models and establish HolmHz's performance floor.**

**Acceptance Criteria:**

- [ ] Phase 1: Freeze backbone, train head only → AUC ≥ 0.80 on val set
- [ ] Phase 2: Unfreeze, fine-tune full model → AUC ≥ 0.85 on val set
- [ ] Hyperparameter search: ≥3 LR values, ≥2 batch sizes tested
- [ ] Best checkpoint saved (`.pt` file) + **resume tested** (disconnect simulation)
- [ ] Training curve (loss, AUC) logged on W&B — no overfitting (val loss ≤ 1.2x train loss)
- [ ] Quick smoke test on 5 ảnh từ `imgs/` folder (Gemini, Flux, Real)
- [ ] Training time documented (epochs, wall clock on Kaggle T4)

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

> **Ước tính thời gian Kaggle T4 (Cập nhật 24/02/2026):**
>
> - ~18k images, batch=32 → ~560 steps/epoch
> - Freeze: ~2 min/epoch × 10 = 20 min
> - Fine-tune: ~10 min/epoch × 20 = 3.3 hours
> - HP search (3 runs): ~10 hours
> - Tổng: ~14 hours ≈ nửa tuần Kaggle quota (30h/tuần)
>
> **⚠️ Nếu dùng Colab Free thay Kaggle:**
>
> - Mỗi session ~4h → cần 3-4 sessions
> - LUÔN mount Drive + save checkpoint mỗi epoch
> - Resume: `--resume outputs/checkpoints/last.pt`

> **Exit criteria cho Milestone 1 (ĐIỀU CHỈNH 24/02):**
>
> - AUC ≥ 0.85 in-domain → ✅ Proceed to Sprint 2
> - AUC 0.75-0.85 → Thêm data (FFHQ full, thêm SD generation) + retry
> - AUC < 0.75 → Xem lại pipeline, augmentation, có thể bug
> - Giảm target từ 0.88→0.85 do dataset nhỏ hơn (18k vs 45k)
