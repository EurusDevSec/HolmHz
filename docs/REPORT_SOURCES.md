# Danh sách nguồn file tham khảo cho Final Report

> **Mục đích**: Liệt kê tất cả file và thư mục trong project HolmHz được tham khảo/trích xuất dữ liệu để viết `docs/final_report.md`.
>
> **Ngày tạo**: 2026-04-16
> **Tổng số file tham khảo**: 65+

---

## 1. 📊 Benchmark & Kết quả thí nghiệm (Source of Truth cho Chương 4)

| File | Nội dung trích xuất | Chương |
|------|---------------------|--------|
| `outputs/benchmark/v2_benchmark_results.json` | **45+ metrics** cho 7 models (AUC, Accuracy, F1, per-source breakdown) — nguồn chính xác nhất | 4.5 |
| `outputs/evaluation_v9_benchmark/eval_report.json` | Ground truth metrics cho EfficientNet-B0 v9 (ID AUC 0.998, OOD AUC 0.896) | 4.5, 4.6 |
| `outputs/benchmark/v2_comparison/kpi_assessment.md` | Đánh giá KPI (5/5 đạt) | 4.9 |
| `outputs/benchmark/v2_comparison/benchmark_table.md` | Bảng so sánh 7 models | 4.5 |
| `outputs/benchmark/final_benchmark/HolmHz Benchmark.md` | Báo cáo benchmark hoàn chỉnh | 4.5, 4.6 |
| `outputs/benchmark/final_benchmark/benchmark_table_final.md` | Bảng benchmark final | 4.5 |
| `outputs/benchmark/final_benchmark/walkthrough.md.resolved` | Walkthrough phân tích chi tiết | 4.6 |

### Biểu đồ (Hình minh họa)

| File | Mô tả | Dùng ở |
|------|-------|--------|
| `outputs/benchmark/final_benchmark/id_vs_ood_auc.png` | Grouped bar chart: ID vs OOD AUC 7 models | Hình 4.1 |
| `outputs/benchmark/final_benchmark/ood_heatmap.png` | Heatmap per-source OOD performance | Hình 4.2 |
| `outputs/benchmark/final_benchmark/radar_comparison.png` | Radar chart so sánh 4 HolmHz models | Hình 4.3 |
| `outputs/benchmark/v2_comparison/model_comparison_bar.png` | Bar chart accuracy by model | Hình 4.4 |
| `outputs/benchmark/v2_comparison/params_vs_ood.png` | Scatter: params vs OOD AUC | Hình 4.5 |
| `outputs/benchmark/v2_comparison/per_source_heatmap.png` | Heatmap per-source cho 4 models | Hình 4.6 |
| `outputs/evaluation_v9_benchmark/roc_curve.png` | ROC curve EfficientNet-B0 v9 | Hình 4.7 |
| `outputs/evaluation_v9_benchmark/confusion_matrix_id.png` | Confusion matrix ID test | Phụ lục |
| `outputs/evaluation_v9_benchmark/confusion_matrix_ood.png` | Confusion matrix OOD test | Phụ lục |
| `outputs/evaluation_v9_benchmark/per_source_accuracy.png` | Per-source accuracy breakdown | Phụ lục |

---

## 2. 📋 Tài liệu dự án (Source cho Mở đầu, KPI, Related Work)

| File | Nội dung trích xuất | Chương |
|------|---------------------|--------|
| `docs/plan.md` | Mục tiêu, KPI, Related Work, Methodology overview | Mở đầu |
| `docs/PROJECT_PLAN.md` | Timeline, milestones, team assignment, sprint plan | Mở đầu, Phụ lục |
| `docs/CONTEXT.md` | Toàn bộ context dự án, decisions, architecture rationale | Tất cả |
| `docs/CRITICAL_ANALYSIS.md` | Phân tích thiếu sót, bài học rút ra | 5.3, 5.4 |
| `docs/CHANGELOG.md` | Lịch sử thay đổi qua các phiên bản | 4.3 |
| `docs/FINAL_REPORT_PLAN.md` | Plan cho báo cáo, outline các chương | Template |
| `README.md` | Project overview, features, tech stack | Mở đầu |

---

## 3. ⚙️ Config huấn luyện (Source cho Chương 3 & 4 — Hyperparameters)

### EfficientNet-B0 configs

| File | Phiên bản | Trọng tâm |
|------|-----------|-----------|
| `configs/train.yaml` | Base config | Default hyperparameters |
| `configs/train_v9.yaml` | **v9 (Best model)** | AdamW, lr=5e-5, cosine, 25 epochs, JPEG aug |
| `configs/train_v2.yaml` | v2 | Dataset v2 switch |
| `configs/train_v3.yaml` | v3 | Augmentation tuning |
| `configs/train_v4.yaml` | v4 | Aggressive augmentation |
| `configs/train_v5.yaml` | v5 | OOD optimization |
| `configs/train_v6.yaml` | v6 | WeightedRandomSampler |

### Multi-architecture configs

| File | Model | Trọng tâm |
|------|-------|-----------|
| `configs/train_resnet18.yaml` | ResNet-18 | Original training config |
| `configs/train_resnet18_v2.yaml` | ResNet-18 v2 | Benchmark config (lr=5e-5, AdamW, 25 epochs) |
| `configs/train_vit_small.yaml` | ViT-Small/16 | Original training config |
| `configs/train_vit_small_v2.yaml` | ViT-Small/16 v2 | Benchmark config (lr=3e-5, AdamW, 30 epochs) |
| `configs/train_swin_tiny.yaml` | Swin-Tiny | Original training config |
| `configs/train_swin_tiny_v2.yaml` | Swin-Tiny v2 | Benchmark config (LR quá cao → training failure) |

### Evaluation & Export configs

| File | Mục đích |
|------|---------|
| `configs/test.yaml` | Evaluation trên test set |
| `configs/test_v9_benchmark.yaml` | Benchmark evaluation cho v9 |
| `configs/test_v11.yaml` | Evaluation cho v11 |
| `configs/export.yaml` | ONNX export settings |

---

## 4. 🧠 Source Code (Source cho Chương 3 — Kiến trúc & Phương pháp)

### Backbones (`src/holmhz/backbones/`)

| File | Class/Function | Thông tin trích xuất |
|------|---------------|---------------------|
| `src/holmhz/backbones/base.py` | `BaseBackbone` | Abstract base class, freeze/unfreeze API |
| `src/holmhz/backbones/efficientnet.py` | `EfficientNetBackbone` | timm create_model, 1280 features_dim, 4M params |
| `src/holmhz/backbones/timm_backbone.py` | `TimmBackbone` | Generic timm wrapper cho ResNet-18, ViT, Swin |

### Detectors (`src/holmhz/detectors/`)

| File | Class/Function | Thông tin trích xuất |
|------|---------------|---------------------|
| `src/holmhz/detectors/base.py` | `BaseDetector` | Abstract detector, predict(), predict_proba() |
| `src/holmhz/detectors/efficientnet_detector.py` | `EfficientNetDetector` | Backbone + Dropout + Linear(1280, 1) |
| `src/holmhz/detectors/timm_detector.py` | `TimmDetector` | Generic detector, _GRADCAM_LAYER_MAP |

### Data Pipeline (`src/holmhz/data/`)

| File | Class/Function | Thông tin trích xuất |
|------|---------------|---------------------|
| `src/holmhz/data/image_dataset.py` | `ImageDataset` | Manifest JSON loading, OpenCV, label counts |
| `src/holmhz/data/transforms.py` | `get_train_transforms()` | JPEG aug (30-100), RandomResizedCrop, ColorJitter |
| `src/holmhz/data/transforms.py` | `get_val_transforms()` | Resize + Normalize only |
| `src/holmhz/data/utils.py` | `create_dataloader()` | DataLoader factory, WeightedRandomSampler |
| `src/holmhz/data/utils.py` | `compute_source_weights()` | Source balancing algorithm |

### Training (`src/holmhz/training/`)

| File | Class/Function | Thông tin trích xuất |
|------|---------------|---------------------|
| `src/holmhz/training/trainer.py` | `Trainer` | Training loop, checkpoint save/load, W&B logging |
| `src/holmhz/training/early_stopping.py` | `EarlyStopping` | patience=7, mode=max (val AUC) |
| `src/holmhz/losses/bce.py` | `get_loss_fn()` | BCEWithLogitsLoss, pos_weight support |
| `src/holmhz/training/schedulers.py` | `get_scheduler()` | CosineAnnealingLR |

### Evaluation (`src/holmhz/evaluation/`)

| File | Class/Function | Thông tin trích xuất |
|------|---------------|---------------------|
| `src/holmhz/evaluation/evaluator.py` | `Evaluator` | Inference loop, per-source breakdown |
| `src/holmhz/metrics/auc.py` | `compute_auc()` | sklearn.metrics.roc_auc_score wrapper |
| `src/holmhz/metrics/accuracy.py` | `compute_accuracy()` | Threshold-based accuracy |
| `src/holmhz/metrics/f1.py` | `compute_f1()` | F1-Score computation |
| `src/holmhz/metrics/precision.py` | `compute_precision()` | Precision computation |
| `src/holmhz/metrics/recall.py` | `compute_recall()` | Recall computation |

### XAI (`src/holmhz/xai/`)

| File | Class/Function | Thông tin trích xuất |
|------|---------------|---------------------|
| `src/holmhz/xai/gradcam.py` | `GradCAMExplainer` | pytorch-grad-cam wrapper, auto target layer |
| `src/holmhz/xai/utils.py` | `load_image_for_gradcam()` | Image preprocessing for Grad-CAM |

### Utils (`src/holmhz/utils/`)

| File | Class/Function | Thông tin trích xuất |
|------|---------------|---------------------|
| `src/holmhz/utils/registry.py` | `Registry`, `DETECTOR_REGISTRY` | Registry Pattern (from DeepfakeBench) |
| `src/holmhz/utils/logger.py` | `get_logger()` | Logging configuration |

---

## 5. 📝 Scripts (Source cho Chương 3 & 4 — Pipeline)

| File | Mục đích | Thông tin trích xuất |
|------|---------|---------------------|
| `scripts/train.py` | Main training entry | Config loading, OmegaConf, W&B init |
| `scripts/test.py` | Evaluation script | Test ID/OOD, eval_report.json generation |
| `scripts/predict.py` | Single image prediction | Inference pipeline, checkpoint loading |
| `scripts/explain.py` | Grad-CAM CLI | XAI gallery generation |
| `scripts/benchmark_sota.py` | Benchmark 4 models | CNNDetection/UniversalFake/DeepfakeBench runners |
| `scripts/prepare_data_v2.py` | Data preparation | 5 Kaggle datasets → manifests |
| `scripts/dataset_stats.py` | Dataset statistics | Train/Val/Test/OOD distribution |
| `scripts/export_onnx.py` | ONNX export | Model conversion for web demo |
| `scripts/generate_xai_gallery.py` | XAI gallery | Batch Grad-CAM heatmap generation |

---

## 6. 🌐 Web Demo (Source cho Chương 3.7)

| File | Mục đích | Thông tin trích xuất |
|------|---------|---------------------|
| `web/app.py` | Gradio UI | Interface layout, predict flow |
| `web/config.py` | Configuration | Model path, device settings |
| `web/model_service.py` | Model service | ONNX inference, preprocessing |

---

## 7. 📁 Dữ liệu (Metadata — không chứa ảnh gốc)

| File/Directory | Nội dung |
|----------------|---------|
| `data/manifests/train.json` | 28.220 training samples manifest |
| `data/manifests/val.json` | 3.526 validation samples manifest |
| `data/manifests/test_id.json` | 3.526 ID test samples manifest |
| `data/manifests/test_ood.json` | 182 OOD test samples manifest |
| `data/manifests/dataset_stats.json` | Summary statistics |
| `data/raw_v2/` | 5 Kaggle dataset folders (ảnh gốc) |

---

## 8. 🧪 Test Suite (Source cho verification)

| File | Nội dung | Xác minh |
|------|---------|---------|
| `tests/test_backbones.py` | EfficientNet backbone tests | 1280 features_dim, 4M params |
| `tests/test_detectors.py` | Detector tests | Forward shape [B,1], freeze/unfreeze, 1281 trainable |
| `tests/test_data.py` | Data pipeline tests | Transform shapes, label dtype, DataLoader |
| `tests/test_evaluator.py` | Evaluator tests | AUC computation, per-source breakdown |
| `tests/test_metrics.py` | Metrics tests | F1, Precision, Recall formulas |
| `tests/test_training.py` | Training tests | Trainer loop, checkpoint format |

---

## 9. 📚 Kaggle Training Docs (Source cho training history)

| File | Nội dung |
|------|---------|
| `docs/KAGGLE_V9_TRAINING.md` | Chi tiết training v9 trên Kaggle T4 GPU |
| `docs/KAGGLE_V6_TRAINING.md` | Training v6 (WeightedRandomSampler) |
| `docs/KAGGLE_V11_TRAINING.md` | Training v11 (CLIP aug experiment) |
| `docs/KAGGLE_MULTI_ARCH_TRAINING.md` | ResNet-18, ViT-Small, Swin-Tiny training |
| `docs/KAGGLE_RETRAIN_3MODELS.md` | Chi tiết retrain 3 models cho benchmark |
| `docs/KAGGLE_TRAINING_V4.md` | Training v4 (dataset v2 first train) |
| `docs/OOD_V5_OPTIMIZATION.md` | OOD optimization journey |
| `docs/STRATEGIC_REORIENTATION.md` | Quyết định chuyển hướng strategy |

---

## 10. 🔧 Infrastructure

| File | Nội dung |
|------|---------|
| `Makefile` | Build targets (train, test, serve, lint) |
| `pyproject.toml` | Project dependencies, package config |
| `.github/workflows/` | CI/CD (lint, test automation) |

---

## Tóm tắt

| Loại | Số lượng file | Vai trò |
|------|--------------|---------|
| Benchmark results | 7 files | Source of Truth cho mọi số liệu |
| Charts/Images | 10 files | Hình minh họa trong báo cáo |
| Project docs | 7 files | Context, KPI, related work |
| Training configs | 19 files | Hyperparameters chính xác |
| Source code | 20+ files | Kiến trúc, pipeline, algorithms |
| Scripts | 9 files | Execution flow |
| Web demo | 3 files | Application layer |
| Data manifests | 5 files | Dataset metadata |
| Tests | 6 files | Verification/cross-check |
| Kaggle docs | 8 files | Training history |
| **Tổng** | **~95 files** | - |

> ⚠️ **Lưu ý**: Tất cả số liệu trong `final_report.md` đều được cross-reference với `outputs/benchmark/v2_benchmark_results.json` (source of truth) và đã xác minh 100% chính xác. Không có số liệu nào được "làm tròn" hoặc "ước lượng" — tất cả là giá trị thực từ evaluation/benchmark trên test set.
