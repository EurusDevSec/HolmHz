## 💡 Context

> **Task ID**: S2-001
> **Phase**: Phase 1 - Data + Model Development
> **Sprint**: Sprint 2 - Evaluation + XAI + Benchmark
> **Status**: ✅ COMPLETED
> **Created**: 10/02/2026
> **Updated**: 26/02/2026 — Evaluation pipeline implemented & run on local RTX 3050
> **Target**: **28/03/2026**
> **Assignee**: Hoàng
> **Blocked by**: S1-006 (trained baseline model) ✅ DONE 26/02 — AUC 0.9983
> **Blocks**: S2-002 (Benchmark cần eval pipeline)
> **Milestone**: M2 — Final model (AUC ≥ 0.90 ID, ≥ 0.75 OOD) + Bảng so sánh + XAI gallery

> Xây dựng evaluation pipeline chuẩn: metrics đa chiều, per-source breakdown,
> OOD evaluation, confusion matrix, ROC curve.
> **ĐẶC BIỆT**: Phân tích vấn đề OOD generalization
> (model dự đoán tất cả ảnh Real camera → FAKE, phát hiện ở smoke test Task 1.6).

---

## 🚨 Vấn đề phát hiện từ Task 1.6 — OOD Generalization Failure

> **Smoke test trên `imgs/Real/` (5 ảnh camera thật):**
>
> ```
> IMG_20211207_152750_319.jpg   FAKE   0.9971
> IMG_20211207_153512_056.jpg   FAKE   0.6556
> IMG_2344.jpg                  FAKE   1.0000
> IMG_2365.jpg                  FAKE   0.9605
> IMG_2369.jpg                  FAKE   0.9924
>
> Summary: 5 images — 5 FAKE, 0 REAL  ← 0% accuracy trên ảnh Real OOD!
> ```
>
> **Root cause — Distribution Shift:**
>
> |                | Training Real                    | imgs/Real/ (smoke test)   |
> | -------------- | -------------------------------- | ------------------------- |
> | Nội dung       | Khuôn mặt (FFHQ, CelebA)         | Ảnh chụp đời thường       |
> | Kích thước gốc | 224×224 (preprocessed)           | 948-4160px (raw camera)   |
> | File size      | 37-97 KB                         | 182-2,736 KB              |
> | Nguồn          | Dataset học thuật (cifake, ffhq) | Ảnh iPhone/camera cá nhân |
>
> Model đã học shortcut: _"ảnh preprocessed từ cifake/ffhq = REAL, khác = FAKE"_
> thay vì features tổng quát. Đây là **distribution shift** điển hình.
>
> **Task 2.1 sẽ**:
>
> 1. Đo lường chính xác mức độ failure trên OOD test set (1,180 ảnh)
> 2. Phân tích per-source: nguồn nào yếu nhất?
> 3. Phân tích error patterns: False Positive vs False Negative
> 4. Ghi nhận kết quả cho báo cáo (limitations section)

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **build a comprehensive evaluation pipeline that computes dataset-level metrics with per-source breakdown on both ID and OOD test sets** so that **I can quantify exactly where the model fails, present statistically meaningful results to the defense committee, and identify targeted improvement areas.**

**Acceptance Criteria:**

- [x] `Evaluator` class: nhận model + dataloader → trả về dict metrics (AUC, Acc, F1, Precision, Recall)
- [x] In-domain evaluation: test trên `test_id.json` (3,975 ảnh) → AUC = 0.9979 ✅
- [x] OOD evaluation: test trên `test_ood.json` (1,180 ảnh) → AUC = 0.4812 ❌ (< 0.75 target)
- [x] Per-source breakdown: 8 sources (cifake, ffhq, stylegan, sd15, flux, tristanzhang_fake, real_pexels, real_camera)
- [x] Visualization: Confusion matrix + ROC curve → saved to `outputs/evaluation/`
- [x] Script `scripts/test.py` chạy end-to-end: load model + eval ID + eval OOD + save report
- [x] JSON report: `outputs/evaluation/eval_report.json` với tất cả metrics
- [x] Phân tích OOD failure: False Positive dominant, real_pexels worst (8.6%)
- [x] Tests: 20 unit tests (11 metrics + 4 evaluator + 5 visualization) — all pass

---

## 🛠️ Implementation

### Subtasks

- [x] 2.1.1 Mở rộng metrics module: thêm `compute_f1()`, `compute_precision()`, `compute_recall()`
- [x] 2.1.2 Implement `src/holmhz/evaluation/evaluator.py` — `Evaluator` class
- [x] 2.1.3 Implement `src/holmhz/utils/visualization.py` — confusion matrix, ROC curve, per-source bar chart
- [x] 2.1.4 Implement `scripts/test.py` — CLI end-to-end
- [x] 2.1.5 Update `configs/test.yaml` — checkpoint path `best.pt`, batch_size=32, num_workers=0
- [x] 2.1.6 Chạy evaluation trên local (RTX 3050, ~1 phút inference)
- [x] 2.1.7 Phân tích kết quả OOD + ghi nhận trong CONTEXT.md (section 15)
- [x] 2.1.8 Unit tests: 20 tests (11 metrics + 4 evaluator + 5 visualization) — all pass

### Branch & PR

- [x] Branch: `feat/s2/evaluation-pipeline`
- [ ] PR Created
- [x] Eval on test_id produces valid metrics — AUC = 0.9979 ✅
- [x] Eval on test_ood produces valid metrics — AUC = 0.4812 (below random, as expected)
- [x] Visualization outputs saved to `outputs/evaluation/` (4 PNG files)
- [x] JSON report saved to `outputs/evaluation/eval_report.json`

---

## 📝 Notes

> **Test data summary (thực tế từ manifests):**
>
> | Set       | Manifest      | Total | Real  | Fake  | Sources                                                         |
> | --------- | ------------- | ----- | ----- | ----- | --------------------------------------------------------------- |
> | In-domain | test_id.json  | 3,975 | 1,797 | 2,178 | cifake(2100), ffhq(750), stylegan(750), sd15(375)               |
> | OOD       | test_ood.json | 1,180 | 600   | 580   | real_pexels(500), real_camera(100), tristanzhang(500), flux(80) |

> **Output format mẫu:**
>
> ```json
> {
>   "model": "efficientnet_b0",
>   "checkpoint": "outputs/checkpoints/best.pt",
>   "timestamp": "2026-02-27T10:00:00",
>   "in_domain": {
>     "manifest": "test_id.json",
>     "total": 3975,
>     "overall": { "auc": 0.998, "accuracy": 0.98, "f1": 0.98 },
>     "per_source": {
>       "cifake": { "accuracy": "?", "n": 2100 },
>       "ffhq": { "accuracy": "?", "n": 750, "note": "real only" },
>       "stylegan": { "accuracy": "?", "n": 750, "note": "fake only" },
>       "sd15": { "accuracy": "?", "n": 375, "note": "fake only" }
>     }
>   },
>   "ood": {
>     "manifest": "test_ood.json",
>     "total": 1180,
>     "overall": { "auc": "?", "accuracy": "?" },
>     "per_source": {
>       "flux": { "accuracy": "?", "n": 80, "label": 1 },
>       "tristanzhang_fake": { "accuracy": "?", "n": 500, "label": 1 },
>       "real_pexels": { "accuracy": "?", "n": 500, "label": 0 },
>       "real_camera": {
>         "accuracy": "?",
>         "n": 100,
>         "label": 0,
>         "note": "likely low"
>       }
>     }
>   }
> }
> ```

> **Kết quả thực tế (26/02/2026):**
>
> - **In-domain AUC**: 0.9979 ✅ (Acc: 0.9814, F1: 0.9830)
> - **OOD AUC**: 0.4812 ❌ (worse than random!)
> - **OOD Fake — flux**: 95.0% accuracy (bias FAKE → đúng trùng hợp)
> - **OOD Fake — tristanzhang_fake**: 87.2% accuracy
> - **OOD Real — real_pexels**: 8.6% accuracy ⚠️ SEVERE
> - **OOD Real — real_camera**: 12.0% accuracy ⚠️ SEVERE
> - **Root cause**: Shortcut learning — model classifies non-face/non-cifake as FAKE
> - **ID-OOD Gap**: 0.5167 — VERY LARGE

> **Files implemented:**
>
> | File                                 | Status | Mô tả                      |
> | ------------------------------------ | ------ | -------------------------- |
> | `src/holmhz/evaluation/evaluator.py` | ✅     | Evaluator class            |
> | `src/holmhz/evaluation/__init__.py`  | ✅     | Exports                    |
> | `src/holmhz/utils/visualization.py`  | ✅     | Confusion matrix, ROC, bar |
> | `scripts/test.py`                    | ✅     | CLI evaluation script      |
> | `src/holmhz/metrics/f1.py`           | ✅     | NEW — compute_f1           |
> | `src/holmhz/metrics/precision.py`    | ✅     | NEW — compute_precision    |
> | `src/holmhz/metrics/recall.py`       | ✅     | NEW — compute_recall       |

> **Metrics đã có (từ Task 1.5) — tái sử dụng:**
>
> - `src/holmhz/metrics/auc.py` → `compute_auc(logits, labels)` ✅
> - `src/holmhz/metrics/accuracy.py` → `compute_accuracy(logits, labels)` ✅
> - Cần thêm: `compute_f1()`, `compute_precision()`, `compute_recall()`

> **Inference chạy trên local (RTX 3050) — không cần Kaggle:**
>
> ```
> test_id.json:  3,975 ảnh × batch=32 → ~124 batches → ~2 min
> test_ood.json: 1,180 ảnh × batch=32 → ~37 batches → ~30s
> Tổng: ~3 phút (chỉ inference, no gradient)
> ```

> **Bài học từ Task 1.6 smoke test:**
>
> 1. Model bias FAKE trên ảnh ngoài distribution
> 2. CIFAKE 32×32 resize lên 224×224 → model có thể đã học artifacts của resize
> 3. FFHQ chỉ là khuôn mặt → model không biết "ảnh Real" ngoài khuôn mặt
> 4. **Task 2.1 phải đo lường chính xác**, không đoán — chạy trên 1,180 OOD ảnh
> 5. Kết quả OOD dù xấu cũng là đóng góp cho báo cáo (limitations analysis)
