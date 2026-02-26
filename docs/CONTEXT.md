# HolmHz Project - Session Context

> File này lưu trữ toàn bộ context của quá trình phát triển dự án để không bị mất giữa các phiên chat.
> Cập nhật lần cuối: 2026-02-25 (sau khi hoàn thành Task 1.3)

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

**Trạng thái**: 35 files tổng. Module `data/` đã có implementation (Task 1.3 ✅). Các module khác EMPTY trừ `__init__.py`.

## 7. Config files

| File                                     | Trạng thái                                      |
| ---------------------------------------- | ----------------------------------------------- |
| `configs/train.yaml`                     | ✅ Có nội dung (model, training, data, wandb)   |
| `configs/test.yaml`                      | ✅ Có nội dung (model, data, evaluation, wandb) |
| `configs/export.yaml`                    | ✅ Có nội dung (model, export ONNX, validation) |
| `configs/detectors/efficientnet_b0.yaml` | ✅ Có nội dung (detector, backbone, head, loss) |

## 8. Tài liệu & files đã tạo

| File                                            | Mô tả                                                                       |
| ----------------------------------------------- | --------------------------------------------------------------------------- |
| `docs/guides/GUIDE_SPRINT1_TASKS.md`            | Hướng dẫn chi tiết Tasks 1.1→1.6 (~2500 dòng), giải thích WHY cho từng bước |
| `docs/guides/GUIDE_TASK_1.2_DATA_COLLECTION.md` | Hướng dẫn chi tiết Task 1.2 (riêng), aligned với plan revised 24/02         |
| `docs/CONTEXT.md`                               | File này — lưu context session                                              |
| `.env.example`                                  | Template biến môi trường (WANDB_API_KEY, DATA_ROOT, DEVICE)                 |
| `docs/DAILY_COMMANDS.md`                        | Các lệnh kiểm tra hàng ngày (lint, test, import, git)                       |
| `notebooks/00_colab_template.ipynb`             | Colab/Kaggle notebook template (7 steps)                                    |
| `Makefile`                                      | Build targets: train, test, serve, lint, format, check, clean               |

## 9. Task Progress

### Sprint 1: Foundation

| Task                       | Trạng thái     | Target (revised)       | Ghi chú                                                             |
| -------------------------- | -------------- | ---------------------- | ------------------------------------------------------------------- |
| **1.1** Environment Setup  | ✅ Completed   | ~~17/02~~ DONE         | Mọi acceptance criteria đã pass                                     |
| **1.2** Data Collection    | ✅ Completed   | **02/03** → DONE 25/02 | 27,680 ảnh processed (26,500 train + 1,180 OOD). ALL CRITERIA PASS  |
| **1.3** Data Pipeline      | ✅ Completed   | **07/03** → DONE 25/02 | 18,550 train / 3,975 val / 3,975 test / 1,180 OOD. 17/17 tests pass |
| **1.4** Model Architecture | ⬜ Not Started | **07/03**              | EfficientNet-B0 backbone + binary head                              |
| **1.5** Training Pipeline  | ⬜ Not Started | **14/03**              | Trainer, loss, metrics, WandB + checkpoint resume                   |
| **1.6** Baseline Training  | ⬜ Not Started | **21/03**              | Train + eval, ưu tiên Kaggle GPU                                    |

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

## 12. Conventions & Lưu ý

- **Luôn dùng đường dẫn đầy đủ**: `.venv/Scripts/python.exe -m pip install ...`
- **Package naming**: PyPI name ≠ import name (vd: `grad-cam` → `pytorch_grad_cam`)
- **Hatchling build**: `packages = ["src/holmhz"]` trong pyproject.toml
- **GPU VRAM 4GB**: Cần batch size nhỏ (8-16), dùng mixed precision (fp16)
- **Background**: Hoàng có kiến thức DevOps, chưa có nền ML/DL → guide cần giải thích concepts
