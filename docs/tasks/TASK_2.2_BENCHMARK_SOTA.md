## 💡 Context

> **Task ID**: S2-002  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 2 - Evaluation + XAI + Benchmark  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Updated**: 03/03/2026  
> **Target**: ~~28/03/2026~~ → **07/04/2026**  
> **Assignee**: Hoàng  
> **Blocked by**: S2-001 (evaluation pipeline) ✅ DONE  
> **Blocks**: Không (nhưng kết quả cần cho báo cáo)  
> **Milestone**: ✅ M2 — Bảng so sánh chính thức + XAI gallery

> Chạy 3 SOTA models đã test ở Phase 0 trên CÙNG test set chuẩn → bảng so sánh chính thức.
> ⚠️ Phase 0 chỉ test 1-2 ảnh → Phase 1 phải test trên dataset level.

---

## 📊 HolmHz Baseline (v4 — Best Model)

> Kết quả chính thức từ Task 1.7 + S2 Threshold Analysis (03/03/2026):

| Metric | ID | OOD | Notes |
| --- | --- | --- | --- |
| **AUC** | **0.9972** | **0.7838** | Target >0.70 ✅ |
| **Accuracy** | 97.3% | 71.2% | threshold=0.76 |
| **F1** | 97.2% | 66.4% | |
| **Precision** | 98.0% | 71.4% | |
| **Recall** | 96.5% | 62.0% | |

**OOD Per-Source (threshold=0.76)**:

| Source | Type | Acc | N | Notes |
| --- | --- | --- | --- | --- |
| flux | Fake | **77.5%** | 80 | Flux.1 schnell |
| tristanzhang_fake | Fake | **79.0%** | 300 | SD+MJ+DALLE mixed |
| real_pexels | Real | **74.5%** | 200 | Pexels outdoor/landscape |
| real_camera | Real | **36.0%** | 100 | Unsplash camera — known limitation |

**Checkpoint**: `outputs/checkpoints/best_v4.pt` (epoch 28, 48.5MB)
**Config**: `configs/test.yaml` (threshold=0.76, v4 checkpoint)
**Training**: EfficientNet-B0, 21,000 samples, WeightedSampler, pos_weight=1.2

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Researcher**, I want to **benchmark all 3 SOTA models (CNNDetection, UniversalFakeDetect, DeepfakeBench) on the exact same test set as HolmHz** so that **I have a fair, apples-to-apples comparison table for the research report.**

**Acceptance Criteria:**

- [ ] CNNDetection chạy trên test set chung → AUC/Acc report
- [ ] UniversalFakeDetect chạy trên test set chung → AUC/Acc report
- [ ] DeepfakeBench (EffNetB4) chạy trên test set chung → AUC/Acc report
- [ ] HolmHz v4 chạy trên test set chung → AUC/Acc report ✅ (đã có)
- [ ] Bảng so sánh chính thức: 4 methods × (AUC ID + AUC OOD + Per-source Acc)
- [ ] ROC curves chồng lên nhau (1 plot, 4 lines)
- [ ] Tất cả kết quả reproducible (script + config committed)

---

## 🛠️ Implementation

### Subtasks

- [ ] 2.2.1 Setup CNNDetection inference trên test set (clone repo + download pretrained)
- [ ] 2.2.2 Setup UniversalFakeDetect inference trên test set (CLIP-based)
- [ ] 2.2.3 Setup DeepfakeBench (EffNetB4) inference trên test set
- [ ] 2.2.4 Script `scripts/benchmark_sota.py` — chạy từng SOTA model trên test set, lưu predictions
- [ ] 2.2.5 Script `analysis/compare_models.py` — tính metrics + ROC overlay + bảng so sánh
- [ ] 2.2.6 Tạo bảng so sánh markdown + ROC overlay plot + per-source heatmap

### Test Set chung

```
ID Test:  data/manifests/test_id.json   (4,545 ảnh)
OOD Test: data/manifests/test_ood.json  (680 ảnh)
Total:    5,225 ảnh

images path: data/processed/train/... (ID) + data/processed/test_ood/... (OOD)
image size: 224×224 PNG
```

### Branch & PR

- [ ] Branch: `feat/s2/benchmark-sota`
- [ ] PR Created
- [ ] Comparison table finalized
- [ ] ROC plot saved

---

## 📝 Notes

> **Test predictions output format:**
>
> ```
> outputs/benchmark/
> ├── holmhz_predictions.csv       ← Đã có (từ test.py)
> ├── cnndetection_predictions.csv
> ├── universalfake_predictions.csv
> ├── deepfakebench_predictions.csv
> └── comparison/
>     ├── comparison_table.md
>     ├── roc_overlay.png
>     └── per_source_heatmap.png
> ```

> **Predictions CSV format:**
>
> ```csv
> image_path,label,source,prob_fake
> data/processed/train/real/cifake/train_0001.png,0,cifake,0.12
> data/processed/test_ood/flux/00001.png,1,flux,0.94
> ```

> **Challenges dự kiến:**
>
> - CNNDetection: resize input 224x224, ImageNet normalization → straightforward
> - UniversalFakeDetect: CLIP normalization (khác ImageNet) → phải dùng đúng transform
> - DeepfakeBench: có thể cần Docker (dlib dependency) → fallback: bỏ face alignment, ghi chú limitation

> **Lưu ý cho báo cáo:**
> Ghi rõ conditions: resolution, normalization, face cropping (nếu có), để hội đồng hiểu comparison là fair.

---

## 📊 Expected Comparison Table (template)

| Model | Type | Train Data | ID AUC | OOD AUC | flux Acc | tristan Acc | real_pexels Acc | real_camera Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **HolmHz v4** | EffNet-B0 | 21K mixed | **0.9972** | **0.7838** | 77.5% | 79.0% | 74.5% | 36.0% |
| CNNDetection | ResNet-50 | ProGAN only | ? | ? | ? | ? | ? | ? |
| UniversalFakeDetect | CLIP-ViT | GAN only | ? | ? | ? | ? | ? | ? |
| DeepfakeBench | EffNet-B4 | FF++ faces | ? | ? | ? | ? | ? | ? |

> **Dự đoán**: SOTA models sẽ kém hơn HolmHz trên OOD Diffusion (flux, tristanzhang) vì chỉ train trên GAN/Face data. Nhưng có thể tốt hơn trên real faces.
