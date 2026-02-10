## 💡 Context

> **Task ID**: S2-001  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 2 - Evaluation + XAI + Benchmark  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: 21/03/2026  
> **Assignee**: Hoàng  
> **Blocked by**: S1-006 (trained baseline model)  
> **Blocks**: S2-002 (Benchmark cần eval pipeline)

> Xây dựng evaluation pipeline chuẩn: AUC, Accuracy, F1, confusion matrix, ROC curve.
> Per-source breakdown (GAN vs Diffusion vs Real) + OOD evaluation riêng.

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **build a comprehensive evaluation pipeline that computes dataset-level metrics with per-source breakdown** so that **I have statistically meaningful results (not just 1-2 image tests) that I can present to the defense committee.**

**Acceptance Criteria:**

- [ ] `Evaluator` class: nhận model + dataloader → trả về dict metrics
- [ ] Metrics: AUC-ROC, Accuracy, F1-Score, Precision, Recall
- [ ] Per-source breakdown: riêng GAN (StyleGAN2, ProGAN), riêng Diffusion (SD, GenImage), riêng Real
- [ ] OOD evaluation riêng: Gemini, Flux, SDXL — **thước đo quan trọng nhất**
- [ ] Visualization: Confusion matrix image, ROC curve image (saved to outputs/)
- [ ] Script `scripts/test.py` chạy end-to-end
- [ ] Output: JSON report + images

---

## 🛠️ Implementation

### Subtasks

- [ ] 2.1.1 Implement `src/holmhz/evaluation/evaluator.py` (metrics computation)
- [ ] 2.1.2 Implement `src/holmhz/metrics/auc.py` + `accuracy.py`
- [ ] 2.1.3 Implement `src/holmhz/utils/visualization.py` (confusion matrix, ROC plot)
- [ ] 2.1.4 Implement `scripts/test.py` CLI
- [ ] 2.1.5 Per-source breakdown logic (group by `source` field in manifest)

### Branch & PR

- [ ] Branch: `feat/s2/evaluation-pipeline`
- [ ] PR Created
- [ ] Eval on val set produces valid metrics
- [ ] Visualization outputs saved

---

## 📝 Notes

> **Output format mẫu:**
>
> ```json
> {
>   "overall": { "auc": 0.91, "accuracy": 0.88, "f1": 0.87 },
>   "per_source": {
>     "ffhq": { "auc": 0.95, "n": 2000, "type": "real" },
>     "stylegan2": { "auc": 0.93, "n": 2000, "type": "gan" },
>     "genimage_sd15": { "auc": 0.89, "n": 2000, "type": "diffusion" }
>   },
>   "ood": {
>     "gemini": { "auc": 0.72, "n": 300, "type": "diffusion_ood" },
>     "flux": { "auc": 0.68, "n": 300, "type": "diffusion_ood" }
>   }
> }
> ```

> **Bài học từ Phase 0:**
>
> - Chạy trên 1-2 ảnh KHÔNG có ý nghĩa thống kê → phải dataset-level
> - Cần ≥500 ảnh per source để AUC có ý nghĩa (p < 0.05)
> - Per-source breakdown cho thấy model mạnh/yếu ở đâu
