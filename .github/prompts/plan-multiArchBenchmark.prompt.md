## Plan: Multi-Architecture Benchmark + Tasks 2.3 & 2.4

Mở rộng HolmHz với 3 backbone mới (ResNet-18, ViT-Small, Swin-T) → retrain trên cùng 21K dataset → so sánh fair với EfficientNet-B0 (v4) → rồi hoàn thành Task 2.3 (Grad-CAM) và 2.4 (ONNX Export). Tận dụng pattern `BaseBackbone`/`BaseDetector`/`Registry` hiện tại — **zero thay đổi code cũ**.

---

### GitNexus Analysis Summary

- **Impact**: LOW risk — `EfficientNetDetector` và `EfficientNetBackbone` không có upstream dependents
- **Pattern**: Clean `timm.create_model()` → `BaseBackbone.extract_features()` → `BaseDetector.forward()` → `DETECTOR_REGISTRY.build()`
- **Metric pipeline**: `compute_auc()`, `compute_accuracy()` in `src/holmhz/metrics/` → `Trainer._log_epoch()` → `Trainer._log_wandb()`
- **Benchmark pipeline**: `scripts/benchmark_sota.py` outputs CSV → `analysis/compare_models.py` reads CSV → comparison table + ROC plots

---

### Phase 1: Architecture Extension (Task 2.2b)

1. **Create generic `TimmBackbone`** — `src/holmhz/backbones/timm_backbone.py` _(new)_ — wraps any `timm` model, reuses `BaseBackbone` interface (`extract_features`, `get_features_dim`, `freeze/unfreeze`)
2. **Create generic `TimmDetector`** — `src/holmhz/detectors/timm_detector.py` _(new)_ — same pattern as `EfficientNetDetector`: Backbone → Dropout(0.3) → Linear(features_dim, 1). Implement `get_feature_layer()` for Grad-CAM compatibility
3. **Register 3 detectors** in `src/holmhz/detectors/__init__.py` — add `resnet18`, `vit_small`, `swin_tiny` to `DETECTOR_REGISTRY` _(3 lines added, existing code untouched)_
4. **Create detector YAML configs** — `configs/detectors/resnet18.yaml`, `vit_small.yaml`, `swin_tiny.yaml` _(parallel with step 5)_
5. **Create training YAML configs** — `configs/train_resnet18.yaml`, `train_vit_small.yaml`, `train_swin_tiny.yaml` — based on `configs/train_v4.yaml` (WeightedSampler, pos_weight=1.2, cosine schedule)
6. **Add unit tests** — update `tests/test_detectors.py` for forward pass shape, freeze/unfreeze, registry lookup

### VRAM Analysis

| Model                | Params | features_dim | batch_size (Kaggle T4 16GB) | batch_size (RTX 3050 4GB) |
| -------------------- | ------ | ------------ | --------------------------- | ------------------------- |
| EfficientNet-B0 (v4) | 4M     | 1280         | 32                          | 32                        |
| **ResNet-18**        | 11M    | 512          | 32                          | 16-32                     |
| **ViT-Small/16**     | 22M    | 384          | 16                          | 8                         |
| **Swin-T**           | 28M    | 768          | 16                          | 8                         |

---

### Phase 2: Training _(depends on Phase 1)_

7. **Train ResNet-18** — `python scripts/train.py configs/train_resnet18.yaml` → `best_resnet18.pt`
8. **Train Swin-T** — `python scripts/train.py configs/train_swin_tiny.yaml` → `best_swin_tiny.pt` _(parallel with 7)_
9. **Train ViT-Small** — `python scripts/train.py configs/train_vit_small.yaml` → `best_vit_small.pt` _(parallel with 7,8)_

Training environment: **Kaggle T4 16GB** (same as v4). Same hyperparams for fair comparison.

---

### Phase 3: Evaluation & Comparison _(depends on Phase 2)_

10. **Evaluate** each model with `scripts/test.py` on same test set (4,545 ID + 680 OOD)
11. **Update** `scripts/benchmark_sota.py` — add runners `run_resnet18`, `run_vit_small`, `run_swin_tiny` following `run_holmhz()` pattern
12. **Generate comparison** — `python analysis/compare_models.py` → updated table with 7 models (4 external SOTA + 3 new internal backbone variants)

---

### Phase 4: Task 2.3 — Grad-CAM XAI _(parallel with Phase 3)_

13. **Implement** `src/holmhz/xai/gradcam.py` _(currently empty)_ — wrapper around `pytorch_grad_cam.GradCAM`, using `model.get_feature_layer()` target
14. **Implement** `src/holmhz/xai/utils.py` — heatmap overlay, gallery generation
15. **Create** `scripts/explain.py` — CLI tool: `--image path.png --model efficientnet_b0`
16. **Generate XAI gallery** — 50 samples (25 real + 25 fake) → `outputs/xai_gallery/`. Optional: compare heatmaps across architectures

---

### Phase 5: Task 2.4 — ONNX Export _(parallel with Phase 4)_

17. **Implement** `src/holmhz/exports/onnx_export.py` _(currently empty)_ — `export_to_onnx()` using `configs/export.yaml`
18. **Implement** `src/holmhz/exports/validate.py` — PyTorch vs ONNX diff < 1e-5
19. **Create** `scripts/export_onnx.py` CLI + CPU latency benchmark

---

### Relevant Files

**Reference (read-only)**:

- `src/holmhz/backbones/efficientnet.py` — `timm.create_model()` pattern to replicate
- `src/holmhz/detectors/efficientnet_detector.py` — Backbone→Dropout→Linear pattern, `get_feature_layer()`
- `src/holmhz/utils/registry.py` — `Registry.register()`, `Registry.build()`
- `src/holmhz/training/trainer.py` — `Trainer.train_one_epoch()`, `_log_wandb()`
- `configs/train_v4.yaml` — Hyperparams template

**To create (8 new files)**:

- `src/holmhz/backbones/timm_backbone.py`, `src/holmhz/detectors/timm_detector.py`
- `configs/detectors/{resnet18,vit_small,swin_tiny}.yaml`
- `configs/train_{resnet18,vit_small,swin_tiny}.yaml`

**To modify (3 files, minimal)**:

- `src/holmhz/detectors/__init__.py` — 3 registry lines
- `src/holmhz/backbones/__init__.py` — export new backbone
- `tests/test_detectors.py` — add tests for new detectors

---

### Verification

1. `pytest tests/test_detectors.py -v` — all old + new tests pass
2. Dry run: each model trains 1 epoch locally without OOM
3. `python scripts/test.py` generates eval reports for each model
4. `python analysis/compare_models.py` generates 7-model comparison table
5. `gitnexus_detect_changes()` confirms only expected files changed
6. Grad-CAM heatmaps are visually meaningful (not random noise)
7. ONNX validation: max diff < 1e-5

---

### Decisions

- **Generic TimmBackbone/TimmDetector** over 3 separate files — DRY, same `timm` pattern
- **ViT-Small/16** (22M) over ViT-Base (86M) — VRAM constraint on 4GB RTX 3050
- **Kaggle T4** for full training — consistent with v4 training environment
- **Fair benchmark**: Same 21K dataset, same augmentation, same hyperparams, same evaluation
- **Scope**: Multi-architecture + Tasks 2.3/2.4 only. Sprint 3-4 excluded
