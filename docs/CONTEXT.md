# HolmHz Project - Session Context

> File này lưu trữ toàn bộ context của quá trình phát triển dự án để không bị mất giữa các phiên chat.
> Cập nhật lần cuối: 2025-07-14

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
├── __init__.py          # Package root (empty)
├── backbones/           # CNN backbones (EfficientNet-B0)
│   ├── __init__.py
│   ├── base.py
│   └── efficientnet.py
├── data/                # Dataset, transforms, dataloader
│   ├── __init__.py
│   ├── dataset.py
│   ├── loader.py
│   └── transforms.py
├── detectors/           # Detector models
│   ├── __init__.py
│   ├── base.py
│   └── binary_detector.py
├── evaluation/          # Eval pipeline
│   ├── __init__.py
│   ├── evaluator.py
│   └── report.py
├── exports/             # ONNX export
│   ├── __init__.py
│   └── onnx_export.py
├── losses/              # Loss functions
│   ├── __init__.py
│   └── cross_entropy.py
├── metrics/             # AUC, accuracy, etc.
│   ├── __init__.py
│   └── classification.py
├── training/            # Training loop
│   ├── __init__.py
│   └── trainer.py
├── utils/               # Helpers
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   └── registry.py
└── xai/                 # Grad-CAM
    ├── __init__.py
    └── gradcam.py
```

**Trạng thái**: Tất cả 38+ files đều EMPTY — chưa có implementation code nào.

## 7. Config files

| File                                     | Trạng thái |
| ---------------------------------------- | ---------- |
| `configs/train.yaml`                     | Empty      |
| `configs/test.yaml`                      | Empty      |
| `configs/export.yaml`                    | Empty      |
| `configs/detectors/efficientnet_b0.yaml` | Empty      |

## 8. Tài liệu đã tạo

| File                                 | Mô tả                                                                       |
| ------------------------------------ | --------------------------------------------------------------------------- |
| `docs/guides/GUIDE_SPRINT1_TASKS.md` | Hướng dẫn chi tiết Tasks 1.1→1.6 (~1500 dòng), giải thích WHY cho từng bước |
| `docs/CONTEXT.md`                    | File này — lưu context session                                              |

## 9. Task Progress

### Sprint 1: Foundation

| Task                       | Trạng thái     | Ghi chú                                          |
| -------------------------- | -------------- | ------------------------------------------------ |
| **1.1** Environment Setup  | 🟡 In Progress | Dependencies ✅, configs empty, wandb chưa login |
| **1.2** Data Collection    | ⬜ Not Started | Cần thu thập ảnh Real + AI-generated             |
| **1.3** Data Pipeline      | ⬜ Not Started | Dataset class, transforms, dataloader            |
| **1.4** Model Architecture | ⬜ Not Started | EfficientNet-B0 backbone + binary head           |
| **1.5** Training Pipeline  | ⬜ Not Started | Trainer, loss, metrics, WandB logging            |
| **1.6** Baseline Training  | ⬜ Not Started | Train + evaluate first model                     |

### Task 1.1 — Chi tiết remaining items

- [x] Tạo virtual environment
- [x] Cài PyTorch + CUDA
- [x] Cài tất cả runtime + dev dependencies
- [x] Install `holmhz` editable package
- [x] Fix `pyproject.toml` package path
- [ ] Viết nội dung config YAML files (train.yaml, efficientnet_b0.yaml)
- [ ] Viết `src/holmhz/__init__.py` (version, metadata)
- [ ] Tạo `.env.example` (WANDB_API_KEY, DATA_DIR, etc.)
- [ ] `wandb login`
- [ ] `ruff check src/` — verify linting works

## 10. Conventions & Lưu ý

- **Luôn dùng đường dẫn đầy đủ**: `.venv/Scripts/python.exe -m pip install ...`
- **Package naming**: PyPI name ≠ import name (vd: `grad-cam` → `pytorch_grad_cam`)
- **Hatchling build**: `packages = ["src/holmhz"]` trong pyproject.toml
- **GPU VRAM 4GB**: Cần batch size nhỏ (8-16), dùng mixed precision (fp16)
- **Background**: Hoàng có kiến thức DevOps, chưa có nền ML/DL → guide cần giải thích concepts
