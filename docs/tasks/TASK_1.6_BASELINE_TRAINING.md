## 💡 Context

> **Task ID**: S1-006  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 1 - Data + Baseline Training  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Updated**: 26/02/2026 (sau khi Task 1.5 hoàn thành)  
> **Target**: **21/03/2026**  
> **Assignee**: Hoàng  
> **Blocked by**: S1-005 (training pipeline) ✅ DONE 26/02  
> **Blocks**: S2-001 (Evaluation cần trained model)  
> **Milestone**: ✅ M1 — Dataset v1 (≥15k) + Baseline AUC ≥ 0.85 (in-domain)

> Train EfficientNet-B0 baseline trên dataset GAN + Diffusion.
> Transfer learning: freeze backbone → train head → unfreeze → fine-tune.

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **train the EfficientNet-B0 baseline model on a mixed GAN + Diffusion dataset** so that **I get a working detector to benchmark against the 3 SOTA models and establish HolmHz's performance floor.**

**Acceptance Criteria:**

- [ ] Phase 1 (Freeze): Train head only, `freeze_backbone=true`, epochs=10 → **AUC ≥ 0.90** on val set
- [ ] Phase 2 (Fine-tune): Unfreeze backbone, `freeze_backbone=false`, LR=1e-4, epochs=20 → **AUC ≥ 0.93** on val set
- [ ] Hyperparameter search: ≥3 LR values tested (Phase 2)
- [ ] Best checkpoint saved (`.pt` file) + **resume tested** (disconnect simulation)
- [ ] Training curve (loss, AUC) logged on W&B — no overfitting (val loss ≤ 1.2× train loss)
- [ ] Quick smoke test trên 10 ảnh từ `imgs/` folder (5 Fake AI-gen + 5 Real)
- [ ] Training time documented (epochs, wall clock trên Kaggle T4 hoặc local RTX 3050)

> **Cập nhật 26/02**: Nâng target AUC dựa trên kết quả dry run Task 1.5:
>
> - Dry run 2 epochs, freeze, batch=8 trên RTX 3050 → Val AUC **0.92** (rất khả quan!)
> - Phase 1 (10 epochs) chắc chắn đạt 0.90+ dựa trên trajectory
> - Phase 2 (unfreeze toàn bộ 4M params) nên đạt ≥0.93

---

## 🛠️ Implementation

### Subtasks

- [ ] 1.6.1 Phase 1 — Freeze backbone, train head only (LR=1e-3, epochs=10, ~20 min)
- [ ] 1.6.2 Phase 2 — Unfreeze backbone, fine-tune full model (LR=1e-4, epochs=20, ~3h)
- [ ] 1.6.3 Hyperparameter tuning Phase 2 (LR: {5e-4, 1e-4, 5e-5}, ≥3 W&B runs)
- [ ] 1.6.4 Implement `scripts/predict.py` — smoke test trên `imgs/` (5 Fake + 5 Real)
- [ ] 1.6.5 Save final best checkpoint + document results in CONTEXT.md

### Branch & PR

- [ ] Branch: `feat/s1/baseline-training`
- [ ] PR Created
- [ ] W&B experiment link attached
- [ ] Best AUC documented

---

## 📝 Notes

> **Kết quả dry run Task 1.5 (26/02/2026) — Cơ sở thực tế:**
>
> | Epoch | Train Loss | Val Loss | Val Acc | Val AUC | LR       | Time |
> | ----- | ---------- | -------- | ------- | ------- | -------- | ---- |
> | 1     | 0.5194     | 0.3986   | 0.8360  | 0.9115  | 5.01e-04 | 117s |
> | 2     | 0.4860     | 0.3867   | 0.8425  | 0.9189  | 1.00e-06 | 116s |
>
> → Freeze backbone, chỉ train **1,281 params** (head), AUC đã 0.92 sau 2 epoch!
> → 10 epochs sẽ converge mạnh hơn, kỳ vọng AUC 0.93-0.95 (freeze).
> → Unfreeze toàn bộ 4,008,829 params sẽ học features sâu hơn.

> **Training Strategy (2 Phases):**
>
> ```
> ┌─────────────── PHASE 1: FREEZE BACKBONE ─────────────────┐
> │  • freeze_backbone = true (default config)                │
> │  • Trainable: 1,281 params (chỉ head)                     │
> │  • LR = 1e-3 (cao, vì ít params)                          │
> │  • Epochs: 10                                             │
> │  • Thời gian: ~20 min (RTX 3050 batch=8)                  │
> │  • Mục đích: Head học phân biệt Real vs Fake nhanh        │
> │  • Kỳ vọng: Val AUC ≥ 0.90                                │
> └───────────────────────────────────────────────────────────┘
>                         │
>                    Load best.pt weights
>                         │
>                         ▼
> ┌─────────────── PHASE 2: FINE-TUNE FULL ─────────────────┐
> │  • freeze_backbone = false                                │
> │  • Trainable: 4,008,829 params (toàn bộ)                  │
> │  • LR = 1e-4 (thấp hơn 10×, tránh phá pretrained)        │
> │  • Epochs: 20                                             │
> │  • Thời gian: ~3h (RTX 3050 batch=8)                      │
> │  • Mục đích: Backbone học artifacts riêng cho deepfake     │
> │  • Kỳ vọng: Val AUC ≥ 0.93                                │
> └───────────────────────────────────────────────────────────┘
> ```

> **CLI Commands (từ infrastructure Task 1.5):**
>
> ```bash
> # Phase 1: Freeze backbone (dùng default config)
> python scripts/train.py training.epochs=10
>
> # Phase 2: Unfreeze + fine-tune
> python scripts/train.py model.freeze_backbone=false \
>     training.learning_rate=0.0001 training.epochs=20
>
> # Local dev (batch nhỏ vì RTX 3050 4GB VRAM)
> python scripts/train.py training.epochs=10 \
>     training.batch_size=8 data.num_workers=0
> ```

> **Hyperparameter Grid (Phase 2 only):**
>
> ```
> LR:          [5e-4, 1e-4, 5e-5]      ← 3 giá trị
> Batch:       32 (Kaggle) / 8 (local)  ← giữ cố định theo GPU
> Scheduler:   CosineAnnealing (T_max = epochs)
> Optimizer:   AdamW (weight_decay=1e-4)
> Early Stop:  patience=5, monitor=val_auc
> ```

> **Ước tính thời gian thực tế (từ dry run 26/02):**
>
> | Setup                | Per epoch (freeze) | Per epoch (unfreeze) | Phase 1 (10ep) | Phase 2 (20ep) |
> | -------------------- | ------------------ | -------------------- | -------------- | -------------- |
> | RTX 3050 (batch=8)   | ~117s              | ~500s (ước tính)     | ~20 min        | ~2.8h          |
> | Kaggle T4 (batch=32) | ~60s               | ~240s (ước tính)     | ~10 min        | ~1.3h          |
>
> Tổng cho HP search (3 runs Phase 2): ~4-8h tùy GPU.

> **W&B Dashboard (đã verified hoạt động 26/02):**
>
> - Project: `holmhz`
> - Entity: `hoangslevan-thu-dau-mot-university`
> - URL: https://wandb.ai/hoangslevan-thu-dau-mot-university/holmhz
> - `.env` chứa `WANDB_API_KEY` (gitignored, auto-loaded by `python-dotenv`)

> **Lưu ý quan trọng cho Phase 2 (unfreeze):**
>
> 1. **Xóa `outputs/checkpoints/last.pt`** trước khi chạy Phase 2 (tránh auto-resume với optimizer cũ chỉ có head params)
> 2. Copy `best.pt` → `phase1_best.pt` rồi xóa cả `best.pt` + `last.pt`
> 3. Model sẽ tự load pretrained ImageNet weights (vì `pretrained=true`)
> 4. LR thấp hơn 10× (1e-4 thay 1e-3) để **không phá pretrained features**
> 5. Nếu AUC giảm sau unfreeze → LR quá cao, thử 5e-5

> **Exit Criteria cho Milestone 1 (Cập nhật 26/02):**
>
> - AUC ≥ 0.93 → ✅ Vượt target, proceed to Sprint 2
> - AUC 0.85-0.93 → ✅ Đạt target, proceed to Sprint 2
> - AUC 0.75-0.85 → Thêm data (FFHQ full 52K) + retry
> - AUC < 0.75 → Xem lại pipeline, có thể bug
>
> **Nâng target từ 0.85 → 0.93** cho Phase 2 dựa trên dry run đã 0.92 (freeze, 2 epoch)

> **Smoke test ảnh `imgs/` (cần implement `scripts/predict.py`):**
>
> ```
> imgs/
> ├── Fake_AI_generated/    # 5 ảnh (Gemini, Flux)
> └── Real/                 # 5 ảnh (camera thật)
> ```
>
> - Input: 1 ảnh → Output: P(Fake), label (Real/Fake)
> - Kiểm tra: Model dự đoán đúng ≥3/5 Fake và ≥3/5 Real
