## 💡 Context

> **Task ID**: S2-002  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 2 - Evaluation + XAI + Benchmark  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: ~~28/03/2026~~ → **07/04/2026**  
> **Assignee**: Hoàng  
> **Blocked by**: S2-001 (evaluation pipeline), S1-003 (shared test set)  
> **Blocks**: Không (nhưng kết quả cần cho báo cáo)  
> **Milestone**: ✅ M2 — Bảng so sánh chính thức + XAI gallery

> Chạy 3 SOTA models đã test ở Phase 0 trên CÙNG test set chuẩn → bảng so sánh chính thức.
> ⚠️ Phase 0 chỉ test 1-2 ảnh → Phase 1 phải test trên dataset level.

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **benchmark all 3 SOTA models (CNNDetection, UniversalFakeDetect, DeepfakeBench) on the exact same test set as HolmHz** so that **I have a fair, apples-to-apples comparison table for the research report.**

**Acceptance Criteria:**

- [ ] CNNDetection chạy trên test set chung → AUC/Acc report
- [ ] UniversalFakeDetect chạy trên test set chung → AUC/Acc report
- [ ] DeepfakeBench (EffNetB4) chạy trên test set chung → AUC/Acc report
- [ ] HolmHz baseline chạy trên test set chung → AUC/Acc report
- [ ] Bảng so sánh chính thức: 4 methods × (AUC ID + AUC OOD GAN + AUC OOD Diffusion)
- [ ] ROC curves chồng lên nhau (1 plot, 4 lines)
- [ ] Tất cả kết quả reproducible (script + config committed)

---

## 🛠️ Implementation

### Subtasks

- [ ] 2.2.1 Setup CNNDetection inference trên test set (Docker hoặc script)
- [ ] 2.2.2 Setup UniversalFakeDetect inference trên test set
- [ ] 2.2.3 Setup DeepfakeBench (EffNetB4) inference trên test set
- [ ] 2.2.4 Script `analysis/compute_auc.py` tính AUC từ predictions CSV
- [ ] 2.2.5 Tạo bảng so sánh markdown + ROC overlay plot

### Branch & PR

- [ ] Branch: `feat/s2/benchmark-sota`
- [ ] PR Created
- [ ] Comparison table finalized
- [ ] ROC plot saved

---

## 📝 Notes

> **Test set format chung:**
> Tất cả 4 models phải chạy trên cùng ảnh → predictions lưu thành CSV:
>
> ```csv
> image_path,label,source,cnndetection_prob,universalfake_prob,deepfakebench_prob,holmhz_prob
> test/real/001.png,0,ffhq,0.12,0.08,0.19,0.15
> test/fake/001.png,1,genimage,0.94,0.87,0.52,0.91
> ```

> **Challenges dự kiến:**
>
> - CNNDetection: resize input 224x224, ImageNet normalization → straightforward
> - UniversalFakeDetect: CLIP normalization (khác ImageNet) → phải dùng đúng transform
> - DeepfakeBench: có thể cần Docker (dlib dependency) → fallback: bỏ face alignment, ghi chú limitation

> **Lưu ý cho báo cáo:**
> Ghi rõ conditions: resolution, normalization, face cropping (nếu có), để hội đồng hiểu comparison là fair.
