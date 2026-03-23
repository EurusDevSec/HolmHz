# Multi-Architecture Benchmark — Implementation Guide

Tài liệu ghi lại toàn bộ quá trình mở rộng HolmHz với 3 backbone mới + Task 2.3 (Grad-CAM) + Task 2.4 (ONNX Export).

---

## 1. Tổng quan

### Mục tiêu

- Thêm 3 backbone mới: **ResNet-18** (baseline), **ViT-Small/16** (Transformer), **Swin-T** (Swin Transformer)
- Retrain trên cùng 21K dataset, cùng hyperparams → fair comparison với EfficientNet-B0 (v4)
- Implement Grad-CAM (giải thích model) và ONNX Export (deployment)

### Kiến trúc mới

| Model                | Params | features_dim | batch_size (T4)  | Kiểu        |
| -------------------- | ------ | ------------ | ---------------- | ----------- |
| EfficientNet-B0 (v4) | 4M     | 1280         | 32               | CNN         |
| **ResNet-18**        | 11M    | 512          | 32               | CNN         |
| **ViT-Small/16**     | 22M    | 384          | 16               | Transformer |
| **Swin-T**           | 28M    | 768          | Swin Transformer |

---

## 2. Phase 1: Architecture Extension

### 2.1 Generic TimmBackbone (`src/holmhz/backbones/timm_backbone.py`)

**Vấn đề**: Mỗi backbone mới nếu tạo file riêng sẽ duplicate 90% code.

**Giải pháp**: Generic `TimmBackbone` wrapping bất kỳ timm model nào.

```python
class TimmBackbone(BaseBackbone):
    def __init__(self, model_name: str, pretrained: bool = True):
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self._features_dim = self.model.num_features  # Auto-detect!
```

Key design decisions:

- `num_classes=0` → timm bỏ classification head, chỉ trả feature vector
- `model.num_features` → tự lấy features_dim, không hardcode
- Kế thừa `BaseBackbone` → `freeze()`, `unfreeze()`, `extract_features()` hoạt động

### 2.2 Generic TimmDetector (`src/holmhz/detectors/timm_detector.py`)

Same pattern như `EfficientNetDetector`:

```
TimmBackbone(model_name) → Dropout(p) → Linear(features_dim, 1)
```

Thêm `get_feature_layer()` cho Grad-CAM — map từ model_name → đúng layer:

- `resnet*` → `layer4` (last residual block)
- `vit_*` → `norm` (final LayerNorm)
- `swin_*` → `norm` (final LayerNorm)

### 2.3 Registry

Dùng `functools.partial` để bind `model_name`:

```python
DETECTOR_REGISTRY.register("resnet18")(partial(TimmDetector, model_name="resnet18"))
DETECTOR_REGISTRY.register("vit_small")(partial(TimmDetector, model_name="vit_small_patch16_224"))
DETECTOR_REGISTRY.register("swin_tiny")(partial(TimmDetector, model_name="swin_tiny_patch4_window7_224"))
```

Khi `train.py` gọi `DETECTOR_REGISTRY.build("resnet18", pretrained=True, dropout=0.3, freeze_backbone=False)` → tự động tạo `TimmDetector(model_name="resnet18", pretrained=True, ...)`.

### 2.4 Configs

6 YAML files tạo mới:

- **Detector configs** (`configs/detectors/`): `resnet18.yaml`, `vit_small.yaml`, `swin_tiny.yaml`
- **Training configs** (`configs/`): `train_resnet18.yaml`, `train_vit_small.yaml`, `train_swin_tiny.yaml`

Tất cả dùng cùng hyperparams v4: AdamW, lr=1e-4, cosine, pos_weight=1.2, WeightedSampler, patience=10.
ViT/Swin giảm batch_size xuống 16 (VRAM constraint).

### 2.5 Tests

36 tests pass (17 mới):

- `TestDetectorRegistry::test_registry_new_detectors` — verify 3 tên mới trong registry
- `TestTimmDetector::test_forward_shape` — verify [B, 1] output cho ResNet-18, ViT-Small, Swin-T
- `TestTimmDetector::test_features_dim` — verify 512, 384, 768
- `TestTimmDetector::test_predict_proba_range` — verify [0, 1]
- `TestTimmDetector::test_freeze_backbone` — verify freeze hoạt động
- `TestTimmDetector::test_get_feature_layer` — verify Grad-CAM layer
- `TestTimmDetector::test_registry_build` — verify build qua registry

---

## 3. Phase 2: Training (Kaggle)

Xem file `docs/KAGGLE_MULTI_ARCH_TRAINING.md` cho hướng dẫn chi tiết.

Tóm tắt:

1. Upload code mới lên Kaggle dataset
2. Tạo 3 notebook riêng (1 per model)
3. Chạy train → download checkpoint `.pt` → đặt vào `weights/`

Commands:

```bash
PYTHONPATH=src python scripts/train.py configs/train_resnet18.yaml data.num_workers=4
PYTHONPATH=src python scripts/train.py configs/train_vit_small.yaml data.num_workers=4
PYTHONPATH=src python scripts/train.py configs/train_swin_tiny.yaml data.num_workers=4
```

---

## 4. Phase 3: Evaluation

Sau khi có checkpoints:

```bash
# Evaluate từng model
PYTHONPATH=src python scripts/test.py model.name=resnet18 model.checkpoint=weights/best_resnet18.pt
PYTHONPATH=src python scripts/test.py model.name=vit_small model.checkpoint=weights/best_vit_small.pt
PYTHONPATH=src python scripts/test.py model.name=swin_tiny model.checkpoint=weights/best_swin_tiny.pt

# So sánh tất cả
python analysis/compare_models.py
```

---

## 5. Task 2.3: Grad-CAM XAI

### Files implemented:

- `src/holmhz/xai/gradcam.py` — `GradCAMExplainer` (wrapper pytorch-grad-cam)
- `src/holmhz/xai/utils.py` — `load_image_for_gradcam()`, `create_comparison_grid()`
- `scripts/explain.py` — CLI tool

### Usage:

```bash
# Single image
python scripts/explain.py --image imgs/test.png --model efficientnet_b0 --checkpoint weights/best_model.pt

# Directory
python scripts/explain.py --image-dir imgs/Fake_AI_generated/ --model resnet18 --checkpoint weights/best_resnet18.pt --output outputs/xai_gallery/
```

### Architecture:

```
GradCAMExplainer(model, device)
├── explain(tensor) → heatmap [H,W] float32
├── overlay(tensor, rgb_image) → [H,W,3] uint8
└── save(tensor, rgb_image, path) → saved file
```

---

## 6. Task 2.4: ONNX Export

### Files implemented:

- `src/holmhz/exports/onnx_export.py` — `export_to_onnx()` function
- `src/holmhz/exports/validate.py` — `validate_onnx()` (PyTorch vs ONNX diff check)
- `scripts/export_onnx.py` — CLI tool + CPU latency benchmark

### Usage:

```bash
# Export
python scripts/export_onnx.py configs/export.yaml --model efficientnet_b0 --checkpoint weights/best_model.pt

# Export + benchmark
python scripts/export_onnx.py configs/export.yaml --model resnet18 --checkpoint weights/best_resnet18.pt --benchmark
```

### Features:

- ONNX opset 17, dynamic batch axis
- onnx-simplifier (if installed)
- Validation: max diff < 1e-5 (5 random samples)
- CPU latency benchmark (100 runs, mean/std/p50/p95)

---

## 7. Files Created/Modified

### New files (11):

| File                                    | Purpose                   |
| --------------------------------------- | ------------------------- |
| `src/holmhz/backbones/timm_backbone.py` | Generic timm backbone     |
| `src/holmhz/detectors/timm_detector.py` | Generic timm detector     |
| `configs/detectors/resnet18.yaml`       | ResNet-18 detector config |
| `configs/detectors/vit_small.yaml`      | ViT-Small detector config |
| `configs/detectors/swin_tiny.yaml`      | Swin-T detector config    |
| `configs/train_resnet18.yaml`           | ResNet-18 training config |
| `configs/train_vit_small.yaml`          | ViT-Small training config |
| `configs/train_swin_tiny.yaml`          | Swin-T training config    |
| `src/holmhz/xai/gradcam.py`             | Grad-CAM explainer        |
| `src/holmhz/xai/utils.py`               | XAI utilities             |
| `src/holmhz/exports/onnx_export.py`     | ONNX export               |
| `src/holmhz/exports/validate.py`        | ONNX validation           |
| `scripts/explain.py`                    | Grad-CAM CLI              |
| `scripts/export_onnx.py`                | ONNX export CLI           |

### Modified files (3):

| File                               | Change                                         |
| ---------------------------------- | ---------------------------------------------- |
| `src/holmhz/backbones/__init__.py` | Added TimmBackbone import + 3 registry entries |
| `src/holmhz/detectors/__init__.py` | Added TimmDetector import + 3 registry entries |
| `tests/test_detectors.py`          | Added 17 new tests for TimmDetector            |

### Populated empty `__init__.py`:

| File                             | Change                                |
| -------------------------------- | ------------------------------------- |
| `src/holmhz/xai/__init__.py`     | Exports GradCAMExplainer, utils       |
| `src/holmhz/exports/__init__.py` | Exports export_to_onnx, validate_onnx |

---

## 8. Test Results

```
36 passed — 0 failed
Coverage: timm_backbone.py 100%, timm_detector.py 85%
All old EfficientNet tests unchanged and passing.
```

---

## 9. Next Steps

1. **Upload code to Kaggle** → train 3 models
2. **Download checkpoints** → `weights/best_resnet18.pt`, `best_vit_small.pt`, `best_swin_tiny.pt`
3. **Run evaluation** → compare all 4 internal models + 4 external SOTA
4. **Generate Grad-CAM gallery** → visual explanations
5. **Export ONNX** → deployment-ready models
6. **Update benchmark** → 7-model comparison table
