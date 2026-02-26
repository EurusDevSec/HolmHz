# HolmHz Project - Session Context

> File này lưu trữ toàn bộ context của quá trình phát triển dự án để không bị mất giữa các phiên chat.
> Cập nhật lần cuối: 2026-02-26 (sau khi hoàn thành Task 1.5)

---

## 1. Thông tin dự án

| Mục                 | Chi tiết                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| **Tên dự án**       | HolmHz — Hệ thống Phát hiện Ảnh Tổng hợp (Synthetic Image Detection)                            |
| **Trường**          | Đại học Thủ Dầu Một, Viện Công nghệ Số, 2025-2026                                               |
| **Nhóm**            | Lê Văn Hoàng (trưởng nhóm, code toàn bộ), Ngô Huỳnh Bảo Luân (hỗ trợ)                           |
| **Công nghệ chính** | CNN (EfficientNet-B0) + Grad-CAM (Explainable AI)                                               |
| **Mục tiêu**        | Phân biệt ảnh thật vs ảnh AI-generated (Diffusion models: Flux, Gemini, DALL-E, Midjourney, SD) |

## 2. Môi trường phát triển

| Thành phần          | Phiên bản / Chi tiết                         |
| ------------------- | -------------------------------------------- |
| **OS**              | Windows, terminal MINGW64/Git Bash           |
| **Python**          | 3.12.4                                       |
| **GPU**             | NVIDIA GeForce RTX 3050 Laptop GPU, 4GB VRAM |
| **CUDA Driver**     | 12.8                                         |
| **PyTorch**         | 2.5.1+cu121 (CUDA enabled, verified)         |
| **Virtual env**     | `.venv/` (tạo bởi `uv 0.9.8`)                |
| **Package manager** | pip 26.0.1 (trong venv)                      |
| **Workspace**       | `R:\_Projects\Eurus_Workspace\HolmHz`        |

## 3. Dependencies đã cài đặt (verified)

### Runtime deps

- `torch` 2.5.1+cu121, `torchvision`
- `timm` (pretrained models)
- `albumentations` (data augmentation)
- `opencv-python` (`cv2`)
- `wandb` (experiment tracking)
- `fastapi`, `uvicorn`, `python-multipart` (API)
- `gradio` (UI)
- `omegaconf`, `hydra-core` (config management)
- `onnx`, `onnxruntime` (model export)
- `scipy` (metrics)
- `rich` (console output)
- `pytorch-grad-cam` (XAI - import as `pytorch_grad_cam`)
- `typer` (CLI)
- `pandas`, `tqdm`, `python-dotenv`, `pillow`, `numpy`

### Dev deps

- `pytest`, `pytest-cov` (testing)
- `ruff` (linting/formatting)
- `pre-commit` (git hooks)
- `ipykernel` (Jupyter)
- `matplotlib`, `seaborn` (visualization)
- `scikit-learn` (metrics, analysis)

### Package `holmhz`

- Installed editable: `pip install -e . --no-deps`
- Import: `import holmhz` ✅
- Source: `src/holmhz/`

## 4. Các vấn đề đã giải quyết

### 4.1 PyTorch không import được trong venv

- **Triệu chứng**: `ModuleNotFoundError: No module named 'torch'`
- **Nguyên nhân**: `.venv` tạo bởi `uv` không include `pip`, nên `pip install` cài vào system Python thay vì venv
- **Fix**: `python -m ensurepip --upgrade` → rồi dùng `.venv/Scripts/python.exe -m pip install`
- **Bài học**: Luôn dùng `.venv/Scripts/python.exe -m pip install` thay vì bare `pip install`

### 4.2 `pytorch-grad-cam` package name

- PyPI package name: `grad-cam` (không phải `pytorch-grad-cam`)
- Import name: `pytorch_grad_cam`
- Fix trong pip: `pip install grad-cam`

### 4.3 `holmhz` không import được

- **Nguyên nhân**: `pyproject.toml` có `packages = ["src"]` → import phải là `src.holmhz`
- **Fix**: Đổi thành `packages = ["src/holmhz"]` → import đúng: `import holmhz`

### 4.4 Typos trong config & **init**.py (fix 2026-02-23)

- `__init__.py`: "Detectioin" → "Detection", "dectectors" → "detectors", "trainning" → "training"
- `train.yaml`: "freeze_backbond" → "freeze_backbone", "augmetation" → "augmentation"

### 4.5 `.env.example` chứa API key thật (fix 2026-02-23)

- **Vấn đề**: WANDB_API_KEY thật bị ghi vào `.env.example` → rủi ro lộ key khi commit
- **Fix**: Thay bằng placeholder `your_wandb_key_here`
- **Bài học**: `.env.example` chỉ chứa template, `.env` chứa giá trị thật (đã có trong .gitignore)

### 4.6 `.gitignore` block `.env.example` (fix 2026-02-23)

- **Vấn đề**: `.env.example` bị ignore → không commit được template lên Git
- **Fix**: Xóa `.env.example` khỏi .gitignore, thêm `data/raw/`, `data/processed/`, `weights/`, `outputs/`

### 4.7 ruff config deprecated warnings (fix 2026-02-23)

- **Vấn đề**: `select`/`ignore`/`isort` ở top-level `[tool.ruff]` → deprecated
- **Fix**: Move sang `[tool.ruff.lint]` và `[tool.ruff.lint.isort]`

### 4.8 Albumentations v2.0.8 API breaking changes (fix 2026-02-25)

- **Vấn đề**: `quality_lower`/`quality_upper` và `var_limit` deprecated trong Albumentations v2.0.8
- **Fix**: `quality_range=(60, 100)` thay `quality_lower/quality_upper`, `std_range=(0.01, 0.03)` thay `var_limit`
- **File**: `src/holmhz/data/transforms.py`

## 5. Key Research Findings

> Từ file `docs/research/` — kết quả chạy 3 mô hình SOTA trên dataset mới

### CNNDetection (Wang et al. 2020)

- Train trên ProGAN → generalize tốt cho GAN-based images
- **FAIL hoàn toàn trên Diffusion images** (Gemini, Flux) — accuracy ≈ random (50%)
- Kết luận: Training data quan trọng hơn model architecture

### UniversalFakeDetect (Ojha et al. 2023)

- Dùng CLIP features + nearest neighbor
- Kết quả tốt hơn CNNDetection nhưng vẫn struggles với modern diffusion models
- Observation: Features từ CLIP không capture được artifacts của Diffusion

### DeepfakeBench

- Framework benchmark nhiều detectors
- Confirm: Không model nào trained trên GAN data generalize tốt sang Diffusion
- **Key insight cho HolmHz**: Phải train trên chính Diffusion-generated images

## 6. Cấu trúc source code

```
src/holmhz/
├── __init__.py              # Package root (__version__ = "0.1.0")
├── backbones/               # CNN backbones (EfficientNet-B0)
│   ├── __init__.py
│   ├── base.py
│   └── efficientnet.py
├── data/                    # Dataset, transforms, dataloader
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── image_dataset.py
│   ├── transforms.py
│   └── utils.py
├── detectors/               # Detector models
│   ├── __init__.py
│   ├── base.py
│   └── efficientnet_detector.py
├── evaluation/              # Eval pipeline
│   ├── __init__.py
│   ├── benchmark.py
│   └── evaluator.py
├── exports/                 # ONNX export
│   ├── __init__.py
│   ├── onnx_export.py
│   └── validate.py
├── losses/                  # Loss functions
│   ├── __init__.py
│   └── bce.py
├── metrics/                 # AUC, accuracy, etc.
│   ├── __init__.py
│   ├── accuracy.py
│   └── auc.py
├── training/                # Training loop
│   ├── __init__.py
│   ├── early_stopping.py
│   ├── lr_schedulers.py
│   └── trainer.py
├── utils/                   # Helpers
│   ├── __init__.py
│   ├── io.py
│   ├── logger.py
│   ├── registry.py
│   └── visualization.py
└── xai/                     # Grad-CAM
    ├── __init__.py
    ├── gradcam.py
    └── utils.py
```

**Trạng thái**: 35 files tổng. Module `data/` (Task 1.3 ✅), `backbones/`, `detectors/`, `utils/registry.py` (Task 1.4 ✅), `metrics/`, `losses/`, `training/`, `utils/logger.py` (Task 1.5 ✅) đã có implementation. Các module khác EMPTY.

## 7. Config files

| File                                     | Trạng thái                                      |
| ---------------------------------------- | ----------------------------------------------- |
| `configs/train.yaml`                     | ✅ Có nội dung (model, training, data, wandb)   |
| `configs/test.yaml`                      | ✅ Có nội dung (model, data, evaluation, wandb) |
| `configs/export.yaml`                    | ✅ Có nội dung (model, export ONNX, validation) |
| `configs/detectors/efficientnet_b0.yaml` | ✅ Có nội dung (detector, backbone, head, loss) |

## 8. Tài liệu & files đã tạo

| File                                               | Mô tả                                                                       |
| -------------------------------------------------- | --------------------------------------------------------------------------- |
| `docs/guides/GUIDE_SPRINT1_TASKS.md`               | Hướng dẫn chi tiết Tasks 1.1→1.6 (~2500 dòng), giải thích WHY cho từng bước |
| `docs/guides/GUIDE_TASK_1.2_DATA_COLLECTION.md`    | Hướng dẫn chi tiết Task 1.2 (riêng), aligned với plan revised 24/02         |
| `docs/guides/GUIDE_TASK_1.3_DATA_PIPELINE.md`      | Hướng dẫn chi tiết Task 1.3 Data Pipeline                                   |
| `docs/guides/GUIDE_TASK_1.4_MODEL_ARCHITECTURE.md` | Hướng dẫn chi tiết Task 1.4 Model Architecture                              |
| `docs/guides/GUIDE_TASK_1.5_TRAINING_PIPELINE.md`  | Hướng dẫn chi tiết Task 1.5 Training Pipeline (~1000 dòng)                  |
| `docs/guides/GUIDE_TASK_1.6_BASELINE_TRAINING.md`  | Hướng dẫn chi tiết Task 1.6 Baseline Training (2-phase, HP tuning)          |
| `docs/CONTEXT.md`                                  | File này — lưu context session                                              |
| `.env.example`                                     | Template biến môi trường (WANDB_API_KEY, DATA_ROOT, DEVICE)                 |
| `docs/DAILY_COMMANDS.md`                           | Các lệnh kiểm tra hàng ngày (lint, test, import, git)                       |
| `notebooks/00_colab_template.ipynb`                | Colab/Kaggle notebook template (7 steps)                                    |
| `Makefile`                                         | Build targets: train, test, serve, lint, format, check, clean               |

## 9. Task Progress

### Sprint 1: Foundation

| Task                       | Trạng thái   | Target (revised)       | Ghi chú                                                                 |
| -------------------------- | ------------ | ---------------------- | ----------------------------------------------------------------------- |
| **1.1** Environment Setup  | ✅ Completed | ~~17/02~~ DONE         | Mọi acceptance criteria đã pass                                         |
| **1.2** Data Collection    | ✅ Completed | **02/03** → DONE 25/02 | 27,680 ảnh processed (26,500 train + 1,180 OOD). ALL CRITERIA PASS      |
| **1.3** Data Pipeline      | ✅ Completed | **07/03** → DONE 25/02 | 18,550 train / 3,975 val / 3,975 test / 1,180 OOD. 17/17 tests pass     |
| **1.4** Model Architecture | ✅ Completed | **07/03** → DONE 26/02 | 30/30 tests pass, integration verified. Backbone + Detector + Registry  |
| **1.5** Training Pipeline  | ✅ Completed | **14/03** → DONE 26/02 | 16/16 tests pass, dry run OK (Val AUC 0.92), checkpoint resume verified |
| **1.6** Baseline Training  | ✅ Completed | **21/03** → DONE 26/02 | Kaggle T4: Phase1 AUC 0.9419, Phase2 AUC 0.9983. predict.py implemented |

### Sprint 2: Evaluation

| Task                        | Trạng thái     | Target (revised) | Ghi chú |
| --------------------------- | -------------- | ---------------- | ------- |
| **2.1** Evaluation Pipeline | ⬜ Not Started | **28/03**        |         |
| **2.2** Benchmark SOTA      | ⬜ Not Started | **07/04**        |         |
| **2.3** Grad-CAM XAI        | ⬜ Not Started | **07/04**        |         |
| **2.4** Model Export        | ⬜ Not Started | **07/04**        |         |

### Sprint 3-4: Web + Report

| Task                   | Trạng thái     | Target (revised) | Ghi chú                     |
| ---------------------- | -------------- | ---------------- | --------------------------- |
| **3.1** Backend API    | ⬜ Not Started | **14/04**        | Overlap với Sprint 2        |
| **3.2** Frontend       | ⬜ Not Started | **28/04**        |                             |
| **4.1** Report Writing | ⬜ Not Started | **30/04**        | Luân bắt đầu Ch1-2 từ 29/03 |
| **4.2** Defense Prep   | ⬜ Not Started | **15/05**        | Task cuối cùng              |

### Timeline Revision Note (24/02/2026)

> **3 rủi ro chính đã xử lý:**
>
> 1. **Dataset**: Bỏ GenImage (50GB) → dùng CIFAKE (500MB, Kaggle 1-click) + FFHQ subset + SD v1.5 self-gen
> 2. **GPU**: Ưu tiên Kaggle (30h/tuần, không disconnect) > Colab Free (backup) > Local RTX 3050 (dev only)
> 3. **Timeline**: Dồn 2 tuần, overlap tasks, Luân viết báo cáo song song từ tháng 3
>
> Xem chi tiết: `docs/PROJECT_PLAN.md` Section 12 (Rủi ro & Giải pháp)

### Task 1.1 — Acceptance Criteria

- [x] Cấu trúc `src/holmhz/` theo best practice (10 submodules)
- [x] `pyproject.toml` configured, `pip install -e .` hoạt động
- [x] `.env.example` có placeholder cho dataset paths, wandb key
- [x] YAML config skeleton: train.yaml, test.yaml, export.yaml, efficientnet_b0.yaml
- [x] `wandb login` thành công (verified: `logged_in: True`)
- [x] `Makefile` có target: train, test, serve, lint, format, check, clean
- [x] `ruff check src/` chạy clean — All checks passed! (0 warnings)
- [x] `.gitignore` bao gồm data/, weights/, outputs/
- [x] `import torch; import timm; import holmhz` — all OK
- [x] Colab/Kaggle notebook template (✅ đã tạo `notebooks/00_colab_template.ipynb`)
- [x] Branch `feat/s1/environment-setup` + PR (✅ pushed, PR tạo trên GitHub)

## 10. Data Collection Progress (Task 1.2) — ✅ COMPLETED 25/02/2026

### Tổng quan

- **Tổng cộng**: 27,680 ảnh đã resize về 224×224 trong `data/processed/`
- **Train**: 26,500 ảnh (Real 12K + GAN 5K + Diffusion 9.5K)
- **OOD Test**: 1,180 ảnh (tristanzhang 500 + real_pexels 500 + flux 80 + real_camera 100)
- **Acceptance Criteria**: ALL PASS
- **Validation**: 27,680/27,680 valid (0 corrupt, 0 wrong size)

### Raw Data (data/raw/) — Nguồn gốc

| Folder                         | Nguồn                                                | Số ảnh  | Resolution | Trạng thái |
| ------------------------------ | ---------------------------------------------------- | ------- | ---------- | ---------- |
| `cifake/train/FAKE`            | CIFAKE (Kaggle) - Stable Diffusion v1.4              | 50,000  | 32×32      | ✅         |
| `cifake/train/REAL`            | CIFAKE (Kaggle) - CIFAR-10                           | 50,000  | 32×32      | ✅         |
| `cifake/test/FAKE`             | CIFAKE (Kaggle)                                      | 10,000  | 32×32      | ✅         |
| `cifake/test/REAL`             | CIFAKE (Kaggle)                                      | 10,000  | 32×32      | ✅         |
| `140k_real_and_fake/`          | 140k-real-and-fake (Kaggle) - StyleGAN               | 140,000 | 256×256    | ✅ bonus   |
| `real/ffhq`                    | FFHQ subset (Kaggle mirror)                          | 5,000   | 512×512    | ✅         |
| `real/ffhq_full`               | FFHQ full (52K)                                      | 52,001  | 512×512    | ✅ backup  |
| `real/cifake_subset`           | Random subset CIFAKE Real                            | 7,000   | 32×32      | ✅         |
| `fake_gan/stylegan`            | 140k-real-and-fake subset (StyleGAN faces)           | 5,000   | 256×256    | ✅         |
| `fake_diffusion/cifake_subset` | Random subset CIFAKE Fake                            | 7,000   | 32×32      | ✅         |
| `fake_diffusion/sd15`          | Self-gen (Colab, `runwayml/stable-diffusion-v1-5`)   | 2,500   | 512×512    | ✅         |
| `ood_test/tristanzhang_fake`   | tristanzhang32 test/fake (SD+MJ+DALLE mixed)         | 500     | 1024×1024  | ✅         |
| `ood_test/real_pexels`         | tristanzhang32 test/real (Pexels/Unsplash)           | 500     | ~4480×6272 | ✅         |
| `ood_test/flux`                | HF Inference API (FLUX.1-schnell) + SD v1.5 fallback | 80      | 1024×1024  | ✅         |
| `ood_test/real_camera`         | Unsplash API (portrait/headshot photos)              | 100     | ~400×446   | ✅         |

### Processed Data (data/processed/) — 224×224 PNG

```
data/processed/
├── train/
│   ├── real/
│   │   ├── cifake/          # 7,000 ảnh
│   │   └── ffhq/            # 5,000 ảnh
│   ├── fake_gan/
│   │   └── stylegan/        # 5,000 ảnh
│   └── fake_diffusion/
│       ├── cifake/           # 7,000 ảnh
│       └── sd15/             # 2,500 ảnh
└── ood_test/
    ├── tristanzhang_fake/    # 500 ảnh
    ├── real_pexels/          # 500 ảnh
    ├── flux/                 # 80 ảnh
    └── real_camera/          # 100 ảnh
```

### Acceptance Criteria — TASK 1.2

- [x] ≥6K ảnh Real → **12,000** (cifake 7K + ffhq 5K) ✅
- [x] ≥5K ảnh Diffusion Fake → **9,500** (cifake 7K + sd15 2.5K) ✅
- [x] ≥3K ảnh GAN Fake → **5,000** (stylegan 5K) ✅
- [x] ≥50 ảnh Flux OOD → **80** ✅
- [x] ≥50 ảnh Real camera OOD → **100** (Unsplash portraits) ✅
- [x] Tất cả ảnh resize về 224×224 → **27,680/27,680 valid** ✅
- [x] `data/manifests/dataset_stats.json` → ✅ tồn tại, all_criteria_pass: true
- [x] `validate_dataset.py` → ALL DATA VALID ✅

### Scripts đã tạo (Task 1.2)

| Script                         | Mô tả                                                           |
| ------------------------------ | --------------------------------------------------------------- |
| `scripts/subset_cifake.py`     | Random subset 7K từ CIFAKE (seed=42, reproducible)              |
| `scripts/subset_ffhq.py`       | Random subset 5K từ FFHQ                                        |
| `scripts/subset_stylegan.py`   | Subset 5K StyleGAN từ 140k-real-and-fake                        |
| `scripts/resize_all.py`        | Resize tất cả raw → 224×224 PNG vào data/processed/ (có resume) |
| `scripts/dataset_stats.py`     | Tạo data/manifests/dataset_stats.json + acceptance check        |
| `scripts/validate_dataset.py`  | Kiểm tra corrupt, wrong size, zero bytes                        |
| `scripts/subset_ood_kaggle.py` | Subset tristanzhang_fake + real_pexels (500 mỗi folder)         |

### Quyết định kỹ thuật quan trọng (25/02/2026)

1. **Folder structure**: `data/processed/train/{real,fake_gan,fake_diffusion}/` + `data/processed/ood_test/` — tách rõ train vs OOD test
2. **Flux OOD**: Gemini API deprecated/paid-only → chuyển sang HF Inference API (FLUX.1-schnell) + SD v1.5 fallback. 80 ảnh total.
3. **Real camera OOD**: Dùng Unsplash API (free tier, 50 req/hr) thay vì tự chụp. 100 portrait photos.
4. **tristanzhang32**: Chỉ tải folder `test/` (~12GB/52GB). Subset giữ 500 fake + 500 real.
5. **140k-real-and-fake**: Dataset bonus 140K StyleGAN faces — dùng subset 5K cho fake_gan/stylegan.
6. **CIFAKE 32×32**: Resize lên 224×224 bị pixelated nhưng model vẫn học texture patterns. Nếu AUC thấp → tăng FFHQ + SD v1.5.
7. **Output format**: Tất cả resize thành PNG (lossless) để thống nhất.
8. **Gemini OOD**: KHÔNG có — `imagen-3.0-generate-001` deprecated, `gemini-2.5-flash-image` cần paid billing. Folder `ood_test/gemini/` rỗng.
9. **dalle, midjourney riêng**: KHÔNG có — tristanzhang_fake đã chứa mixed SD+MJ+DALLE.

### Next Step

→ **Task 1.3: Data Pipeline** — Viết code đọc ảnh từ `data/processed/` vào PyTorch DataLoader, implement train/val/test split, augmentation pipeline.

---

## 11. Data Pipeline Progress (Task 1.3) — ✅ COMPLETED 25/02/2026

### Tổng quan

- **Mục tiêu**: Xây dựng data pipeline từ ảnh PNG → PyTorch DataLoader
- **Branch**: `feat/s1/data-pipeline`
- **Tests**: 17/17 passed, 0 warnings (7.7s)

### Data Splits (seed=42, stratified by source, ratio 70/15/15)

| Split       | Total  | Real  | Fake   | File                           |
| ----------- | ------ | ----- | ------ | ------------------------------ |
| **Train**   | 18,550 | 8,427 | 10,123 | `data/manifests/train.json`    |
| **Val**     | 3,975  | 1,776 | 2,199  | `data/manifests/val.json`      |
| **Test ID** | 3,975  | 1,797 | 2,178  | `data/manifests/test_id.json`  |
| **OOD**     | 1,180  | 600   | 580    | `data/manifests/test_ood.json` |

### Files đã implement

| File                                  | Mô tả                                                               |
| ------------------------------------- | ------------------------------------------------------------------- |
| `preprocessing/build_splits.py`       | Script tạo 4 JSON manifests (stratified split)                      |
| `src/holmhz/data/transforms.py`       | `get_train_transforms()`, `get_val_transforms()`                    |
| `src/holmhz/data/image_dataset.py`    | `ImageDataset` class (cv2 + Albumentations)                         |
| `src/holmhz/data/utils.py`            | `create_dataloader()`, `get_dataset_info()`                         |
| `src/holmhz/data/__init__.py`         | Exports: ImageDataset, transforms, utils, constants                 |
| `tests/test_data.py`                  | 17 tests: TestTransforms(5), TestImageDataset(9), TestDataLoader(3) |
| `scripts/verify_pipeline.py`          | Standalone terminal verification script                             |
| `notebooks/01_data_exploration.ipynb` | 6-cell interactive exploration notebook                             |

### DataLoader Output Interface (cho Task 1.4/1.5)

```python
batch = {
    "image": tensor [B, 3, 224, 224],  # float32, normalized ImageNet
    "label": tensor [B],                # float32, 0.0 hoặc 1.0
    "source": list[str],               # ["cifake", "stylegan", ...]
    "path": list[str],                  # ["data/processed/...", ...]
}
```

### Augmentation Pipeline

- **Train**: Resize(224) → HFlip(p=0.5) → OneOf[JPEG(`quality_range=(60,100)`), GBlur, GNoise(`std_range=(0.01,0.03)`)](p=0.3) → ColorJitter(p=0.3) → Normalize(ImageNet) → ToTensorV2
- **Val/Test**: Resize(224) → Normalize(ImageNet) → ToTensorV2
- **ImageNet stats**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Quyết định kỹ thuật

1. **Manifest JSON** thay ImageFolder: reproducible, biết source từng ảnh, hỗ trợ per-source metrics
2. **Stratified split by source**: mỗi source (cifake, ffhq, stylegan, sd15) đúng tỷ lệ 70/15/15
3. **OOD tách riêng hoàn toàn**: Flux, tristanzhang, real_pexels, real_camera KHÔNG lẫn vào train
4. **Binary labels float32**: Tương thích BCEWithLogitsLoss (Task 1.5)
5. **Albumentations v2.0.8**: Dùng `quality_range`, `std_range` (API mới, không deprecated)

---

## 12. Model Architecture Progress (Task 1.4) — ✅ COMPLETED 26/02/2026

### Tổng quan

- **Mục tiêu**: Xây dựng EfficientNet-B0 detector: backbone + head + registry
- **Branch**: `feat/s1/model-architecture`
- **Tests**: 30/30 passed (11 backbone + 19 detector)
- **Integration**: Model nhận batch từ DataLoader, forward pass + loss OK

### Files đã implement

| File                                            | Mô tả                                                       |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `src/holmhz/backbones/base.py`                  | `BaseBackbone` abstract (extract_features, freeze/unfreeze) |
| `src/holmhz/backbones/efficientnet.py`          | `EfficientNetBackbone` wrapping timm, 1280-dim features     |
| `src/holmhz/backbones/__init__.py`              | Exports + BACKBONE_REGISTRY registration                    |
| `src/holmhz/detectors/base.py`                  | `BaseDetector` abstract (forward, predict, predict_proba)   |
| `src/holmhz/detectors/efficientnet_detector.py` | `EfficientNetDetector` = backbone + Dropout(0.3) + Linear   |
| `src/holmhz/detectors/__init__.py`              | Exports + DETECTOR_REGISTRY registration                    |
| `src/holmhz/utils/registry.py`                  | `Registry` class + BACKBONE_REGISTRY + DETECTOR_REGISTRY    |
| `src/holmhz/utils/__init__.py`                  | Exports Registry, registries                                |
| `tests/test_backbones.py`                       | 11 tests: base abstract, features, freeze, params           |
| `tests/test_detectors.py`                       | 19 tests: forward, predict, freeze, gradient, registry      |
| `scripts/check_integrate_data_pipeline.py`      | Integration test: DataLoader → Model → Loss                 |

### Model Interface (cho Task 1.5)

```python
# Tạo model qua Registry (config-driven)
import holmhz.detectors  # trigger registration
model = DETECTOR_REGISTRY.build(
    "efficientnet_b0",
    pretrained=True,
    dropout=0.3,
    freeze_backbone=True,
)

# Forward pass
logits = model(batch["image"])  # [B, 3, 224, 224] → [B, 1]

# Loss (Task 1.5)
loss = BCEWithLogitsLoss(logits.squeeze(1), batch["label"])  # squeeze [B,1]→[B]

# Inference
probs = model.predict_proba(x)  # [B, 1] ∈ [0, 1]
labels = model.predict(x)       # [B, 1] ∈ {0, 1}
```

### Params breakdown

- **Backbone**: 4,007,548 params (EfficientNet-B0)
- **Head**: 1,281 params (Linear(1280, 1) + bias)
- **Total**: 4,008,829 params (~4M, under 6M limit)
- **Freeze backbone**: Trainable = 1,281 (chỉ head)
- **Unfreeze all**: Trainable = 4,008,829 (toàn bộ)

---

## 13. Training Pipeline Progress (Task 1.5) — ✅ COMPLETED 26/02/2026

### Tổng quan

- **Mục tiêu**: Xây dựng training pipeline: Trainer, loss, metrics, scheduler, early stopping, checkpoint
- **Branch**: `feat/s1/trainning-pipeline`
- **Tests**: 16/16 passed (6 metrics + 3 loss + 5 early stopping + 2 scheduler)
- **Dry run**: 2 epochs, batch=8, AMP=True trên RTX 3050 → Val AUC 0.9173
- **Resume**: Checkpoint resume verified (epoch 2 → epoch 3, seamless)
- **All tests**: 63/63 passed (data 17 + backbone 11 + detector 19 + training 16)

### Dry Run Results (2 epochs, batch_size=8, freeze_backbone=True)

| Epoch | Train Loss | Val Loss | Val Acc | Val AUC | LR       | Time   |
| ----- | ---------- | -------- | ------- | ------- | -------- | ------ |
| 1     | 0.5174     | 0.4026   | 0.8327  | 0.9081  | 5.01e-04 | 105.0s |
| 2     | 0.4786     | 0.3835   | 0.8438  | 0.9173  | 1.00e-06 | 111.7s |

> Val AUC 0.91+ sau chỉ 2 epoch (freeze backbone, chỉ train 1,281 params) — rất khả quan!

### Files đã implement

| File                                    | Mô tả                                                              |
| --------------------------------------- | ------------------------------------------------------------------ |
| `src/holmhz/metrics/accuracy.py`        | `compute_accuracy(logits, labels)` — binary accuracy               |
| `src/holmhz/metrics/auc.py`             | `compute_auc(logits, labels)` — AUC via sklearn                    |
| `src/holmhz/metrics/__init__.py`        | Exports compute_accuracy, compute_auc                              |
| `src/holmhz/losses/bce.py`              | `get_loss_fn()` factory — BCEWithLogitsLoss                        |
| `src/holmhz/losses/__init__.py`         | Exports get_loss_fn                                                |
| `src/holmhz/utils/logger.py`            | `get_logger()` — structured logging setup                          |
| `src/holmhz/training/lr_schedulers.py`  | `get_scheduler()` factory — CosineAnnealingLR                      |
| `src/holmhz/training/early_stopping.py` | `EarlyStopping` class — patience, state_dict support               |
| `src/holmhz/training/trainer.py`        | `Trainer` class — train/val loop, AMP, W&B, checkpoint save/resume |
| `src/holmhz/training/__init__.py`       | Exports Trainer, EarlyStopping, get_scheduler                      |
| `scripts/train.py`                      | CLI entry point — OmegaConf config, DETECTOR_REGISTRY.build()      |
| `tests/test_training.py`                | 16 tests: metrics, loss, early stopping, scheduler                 |

### Training Interface (cho Task 1.6)

```python
# Dry run (local, 2 epochs)
python scripts/train.py training.epochs=2 training.batch_size=8 data.num_workers=0

# Full training (Kaggle GPU)
python scripts/train.py

# Resume from checkpoint (auto if last.pt exists)
python scripts/train.py training.epochs=30
```

### Bugs fixed during implementation

1. **CLI arg parsing**: `training.epochs=2` was treated as config path instead of override → fixed detection logic (`= not in arg`)
2. **`total_mem` AttributeError**: PyTorch uses `total_memory` not `total_mem` → fixed
3. **Missing trailing newlines**: 11 files missing EOF newline → auto-fixed by `ruff check --fix`
4. **Import sorting**: `tests/test_training.py` imports unsorted → auto-fixed by ruff

---

## 14. Baseline Training Results (Task 1.6) — ✅ COMPLETED 26/02/2026

### Tổng quan

- **Mục tiêu**: Train EfficientNet-B0 baseline trên 18,550 ảnh (GAN + Diffusion)
- **Branch**: `feat/s1/baseline-training`
- **Platform**: Kaggle T4 x2 (16GB VRAM), batch_size=32, num_workers=4
- **Strategy**: Phase 1 (freeze backbone) → Phase 2 (fine-tune) → HP tuning (3 LR)
- **Best Val AUC**: **0.9983** (Phase 2, LR=1e-4)
- **W&B**: https://wandb.ai/hoangslevan-thu-dau-mot-university/holmhz
- **Total training time**: ~45 min (Kaggle T4)

### Phase 1: Freeze backbone (head only) — run `warm-universe-3`

| Config           | Value                |
| ---------------- | -------------------- |
| freeze_backbone  | true                 |
| trainable params | 1,281                |
| LR               | 1e-3                 |
| Epochs           | 10/10                |
| Batch size       | 32                   |
| Best Val AUC     | **0.9419** (epoch 8) |

### Phase 2: Fine-tune (unfreeze) — run `misunderstood-blaze-4`

| Config           | Value                 |
| ---------------- | --------------------- |
| freeze_backbone  | false                 |
| trainable params | 4,008,829             |
| LR               | 1e-4                  |
| Epochs           | 17/20 (early stop)    |
| Batch size       | 32                    |
| Best Val AUC     | **0.9983** (epoch 12) |

### HP Tuning Results

| Run               | LR   | Best Val AUC | Epochs run | Early Stop | W&B Run               |
| ----------------- | ---- | ------------ | ---------- | ---------- | --------------------- |
| Phase 2 (default) | 1e-4 | **0.9983**   | 17/20      | Ep 17      | misunderstood-blaze-4 |
| HP Run A          | 5e-4 | 0.9982       | 7/20       | Ep 7       | fine-resonance-5      |
| HP Run B          | 5e-5 | 0.9978       | 8/20       | Ep 8       | fanciful-eon-6        |

**Winner: LR=1e-4** (0.9983 AUC) — checkpoint: `hp_lr1e4_best.pt` → `best.pt`

### Kaggle Output Files

| File             | Size    | Description                      |
| ---------------- | ------- | -------------------------------- |
| phase1_best.pt   | 16.3 MB | Phase 1 freeze-only checkpoint   |
| hp_lr1e4_best.pt | 48.5 MB | **Best model** — Phase 2 LR=1e-4 |
| hp_lr5e4_best.pt | 48.5 MB | HP Run A — LR=5e-4               |
| hp_lr5e5_best.pt | 48.5 MB | HP Run B — LR=5e-5               |

### Key Observations

1. **Phase 1 → Phase 2 jump**: AUC 0.9419 → 0.9983 (+0.0564) — unfreezing backbone giúp rất nhiều
2. **All 3 LRs converge**: AUC > 0.997 cho cả 3 → model robust, không nhạy LR
3. **Early stopping effective**: All Phase 2 runs dừng sớm (7-17 epochs), tiết kiệm GPU
4. **Overfitting nhẹ**: Phase 2 train_loss=0.006 vs val_loss=0.065 → gap 10x nhưng AUC vẫn cao
5. **Kaggle T4 rất hiệu quả**: ~1 min/epoch Phase 1, ~1 min/epoch Phase 2

### Milestone 1 Status

- [x] Dataset ≥ 15K: 22,525 ✅
- [x] Baseline AUC ≥ 0.85: **0.9983** ✅ (vượt xa target)
- [x] W&B dashboard có training curves ✅
- [x] Checkpoint saved: `outputs/checkpoints/best.pt` ✅
- [x] predict.py implemented ✅

### Files added/modified

| File                   | Change                                                    |
| ---------------------- | --------------------------------------------------------- |
| `scripts/predict.py`   | NEW — inference script for single/batch images            |
| `pyproject.toml`       | FIX — `pytorch-grad-cam` → `grad-cam` (correct PyPI name) |
| `docs/log_task1.6.txt` | NEW — full Kaggle training log                            |

---

## 15. Conventions & Lưu ý

- **Luôn dùng đường dẫn đầy đủ**: `.venv/Scripts/python.exe -m pip install ...`
- **Package naming**: PyPI name ≠ import name (vd: `grad-cam` → `pytorch_grad_cam`)
- **Hatchling build**: `packages = ["src/holmhz"]` trong pyproject.toml
- **GPU VRAM 4GB**: Cần batch size nhỏ (8-16), dùng mixed precision (fp16)
- **Background**: Hoàng có kiến thức DevOps, chưa có nền ML/DL → guide cần giải thích concepts
