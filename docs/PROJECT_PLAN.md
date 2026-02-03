# 🔍 HolmHz - Synthetic Image Detection System

> **Triển khai và đánh giá các phương pháp CNN cho bài toán phát hiện ảnh tổng hợp**  
> Thời gian: 11/2025 - 05/2026 (7 tháng)  
> Thực hiện: Lê Văn Hoàng (Chính) | Ngô Huỳnh Bảo Luân (Hỗ trợ)  
> Loại hình: Nghiên cứu ứng dụng (Applied Research)
>
> 📌 **Phân công vai trò**:
>
> - **Hoàng**: Toàn bộ kỹ thuật (model, training, code, API)
> - **Luân** (SV năm 1): Hỗ trợ nhẹ (download data, viết báo cáo, test UI)

---

## 📋 Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Định vị nghiên cứu](#2-định-vị-nghiên-cứu)
3. [Dataset Sources](#3-dataset-sources)
4. [Tech Stack](#4-tech-stack)
5. [Kiến trúc hệ thống](#5-kiến-trúc-hệ-thống)
6. [Roadmap & Sprints](#6-roadmap--sprints)
7. [Chi tiết Tasks](#7-chi-tiết-tasks)
8. [KPIs & Metrics](#8-kpis--metrics)
9. [Evaluation Protocol](#9-evaluation-protocol)
10. [Cấu trúc thư mục](#10-cấu-trúc-thư-mục)
11. [Hướng mở rộng](#11-hướng-mở-rộng)

---

## 1. Tổng quan dự án

### 1.1. Bối cảnh & Động lực

Phát hiện ảnh tổng hợp (Synthetic Image Detection) là bài toán đã được nghiên cứu rộng rãi từ 2019 với hàng trăm công bố khoa học. Tuy nhiên:

- **Thách thức hiện tại**: Cross-dataset generalization vẫn chưa được giải quyết triệt để
- **Khoảng trống tại Việt Nam**: Thiếu nghiên cứu và đánh giá trong ngữ cảnh Việt Nam
- **Nhu cầu giáo dục**: Cần công cụ demo để nâng cao nhận thức cộng đồng

### 1.2. Mục tiêu dự án

**Mục tiêu chính**: Triển khai, đánh giá và so sánh các phương pháp CNN hiện đại cho bài toán phát hiện ảnh chân dung tổng hợp.

**Mục tiêu cụ thể**:
| # | Mục tiêu | Đo lường |
|---|----------|----------|
| 1 | Reproduce baseline CNN (EfficientNet-B0) | AUC ≥ 0.90 in-domain |
| 2 | Đánh giá cross-dataset generalization | AUC ≥ 0.75 OOD |
| 3 | So sánh với 3-5 SOTA methods | Bảng comparison |
| 4 | Tích hợp XAI (Grad-CAM) | Heatmap visualization |
| 5 | Web demo proof-of-concept | Latency ≤ 2s/ảnh |

### 1.3. Phạm vi (Scope)

#### ✅ Trong phạm vi (Phase 1 - Ảnh tĩnh)

- Ảnh tĩnh chân dung người (static face images)
- Phát hiện ảnh từ GAN (StyleGAN, ProGAN) và Diffusion (Stable Diffusion)
- Benchmark với các methods có sẵn
- Web demo proof-of-concept

#### 🔜 Mở rộng tương lai (Phase 2 - Video/Audio)

- Video deepfake detection
- Audio deepfake detection
- Real-time processing

#### ❌ Ngoài phạm vi

- Forensic-grade accuracy cho pháp lý
- Commercial deployment at scale
- Adversarial robustness testing

### 1.4. Đóng góp dự kiến (Contributions)

> ⚠️ **Lưu ý**: Đây là nghiên cứu **ứng dụng**, không claim novelty về kiến trúc mới.

| Contribution  | Loại        | Mô tả                                              |
| ------------- | ----------- | -------------------------------------------------- |
| Reproduction  | Engineering | Triển khai lại baseline CNN cho deepfake detection |
| Benchmark     | Evaluation  | So sánh hiệu năng các methods trên dataset chuẩn   |
| XAI Demo      | Application | Tích hợp Grad-CAM với giao diện người dùng         |
| Documentation | Education   | Tài liệu hướng dẫn chi tiết cho người học          |

---

## 2. Định vị nghiên cứu

### 2.1. Prior Art - Các công trình liên quan

| Paper/Project              | Năm  | Phương pháp                  | AUC In-domain | AUC OOD      |
| -------------------------- | ---- | ---------------------------- | ------------- | ------------ |
| Wang et al. (CNNDetection) | 2020 | ResNet50 + blur augmentation | 0.99          | 0.78         |
| Frank et al. (Frequency)   | 2020 | DCT analysis                 | 0.95          | 0.72         |
| UniversalFakeDetect        | 2023 | CLIP features                | 0.95          | 0.82         |
| NPR-DeepfakeDetection      | 2024 | CNN + NPR preprocessing      | 0.97          | 0.84         |
| DeepfakeBench              | 2023 | Benchmark 15+ methods        | -             | -            |
| **HolmHz (Ours)**          | 2026 | EfficientNet-B0 + Grad-CAM   | Target: 0.90  | Target: 0.75 |

### 2.2. Vị trí của dự án

```
                    ┌─────────────────────────────────────────┐
                    │         LANDSCAPE NGHIÊN CỨU            │
                    └─────────────────────────────────────────┘

     Novel Research          Applied Research         Engineering
     (Kiến trúc mới)         (Ứng dụng/Đánh giá)      (Sản phẩm)
           │                        │                      │
           │                        │                      │
    ┌──────┴──────┐          ┌──────┴──────┐        ┌──────┴──────┐
    │ LAA-Net     │          │             │        │ Sensity     │
    │ DRCT        │          │  HolmHz ◄───┼────────│ Deepware    │
    │ UCF         │          │  (Dự án)    │        │ Hive        │
    └─────────────┘          └─────────────┘        └─────────────┘

    Đóng góp:                Đóng góp:               Đóng góp:
    - Kiến trúc mới          - Benchmark             - Sản phẩm thực
    - SOTA performance       - Documentation         - Scalability
    - Publication            - Education             - UX/UI
```

### 2.3. Baseline Methods để so sánh

| Method                     | Source                                                                  | Lý do chọn                   |
| -------------------------- | ----------------------------------------------------------------------- | ---------------------------- |
| **ResNet50 (Wang et al.)** | [GitHub](https://github.com/PeterWang512/CNNDetection)                  | Baseline chuẩn, dễ reproduce |
| **EfficientNet-B0**        | timm library                                                            | Nhẹ, hiệu quả                |
| **CLIP-based**             | [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect) | SOTA generalization          |
| **Frequency (DCT)**        | [GANDCTAnalysis](https://github.com/RUB-SysSec/GANDCTAnalysis)          | Interpretable                |

---

## 3. Dataset Sources

### 3.1. Nguồn dữ liệu công khai

#### A. Ảnh thật (Real Images)

| Dataset         | Số ảnh | Link                                                     | Sử dụng     |
| --------------- | ------ | -------------------------------------------------------- | ----------- |
| **FFHQ**        | 70,000 | [GitHub](https://github.com/NVlabs/ffhq-dataset)         | Train + Val |
| **CelebA-HQ**   | 30,000 | [Link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Train       |
| **DFFD (Real)** | 58,703 | [MSU](http://cvlab.cse.msu.edu/dffd-dataset.html)        | Test        |

#### B. Ảnh GAN (Fake)

| Dataset             | Nguồn                                                             | Số ảnh       | Sử dụng |
| ------------------- | ----------------------------------------------------------------- | ------------ | ------- |
| **DFFD (Fake)**     | Multiple GANs                                                     | 240,336      | Train   |
| **StyleGAN2 Faces** | [NVlabs](https://github.com/NVlabs/stylegan2)                     | Generate 10k | Train   |
| **ProGAN Faces**    | [tkarras](https://github.com/tkarras/progressive_growing_of_gans) | Generate 5k  | Val     |

#### C. Ảnh Diffusion (Fake)

| Dataset                   | Nguồn                                                  | Số ảnh     | Sử dụng          |
| ------------------------- | ------------------------------------------------------ | ---------- | ---------------- |
| **GenImage**              | [GitHub](https://github.com/GenImage-Dataset/GenImage) | Subset 20k | Train + Test OOD |
| **Stable Diffusion v1.5** | Self-generate                                          | 10k        | Train            |
| **SDXL**                  | Self-generate                                          | 5k         | Test OOD         |

### 3.2. Chiến lược chia tập dữ liệu

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET SPLIT STRATEGY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TRAIN (70%)                 VAL (15%)           TEST (15%)     │
│  ─────────────               ─────────           ─────────      │
│  Real:                       Real:               Real:          │
│  • FFHQ (10k)               • FFHQ (2k)         • DFFD (3k)     │
│  • CelebA-HQ (5k)           • CelebA-HQ (1k)                    │
│                                                                 │
│  Fake (GAN):                Fake (GAN):         Fake (GAN):     │
│  • StyleGAN2 (8k)           • ProGAN (2k)       • StarGAN (OOD) │
│  • DFFD-GAN (7k)                                                │
│                                                                 │
│  Fake (Diffusion):          Fake (Diffusion):   Fake (Diff):    │
│  • SD v1.5 (8k)             • SD v2.1 (2k)      • SDXL (OOD)    │
│  • GenImage (7k)                                • MJ proxy (OOD)│
│                                                                 │
│  Total: ~45k                Total: ~7k          Total: ~8k      │
│                                                                 │
│  ⚠️ OOD = Out-of-Distribution (nguồn chưa thấy khi train)      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3. Data Augmentation Strategy

```python
# Training augmentations (mô phỏng điều kiện thực tế)
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.ImageCompression(quality_lower=60, quality_upper=100),  # JPEG
        A.GaussianBlur(blur_limit=(3, 7)),
        A.GaussNoise(var_limit=(10, 50)),
    ], p=0.3),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Validation/Test - no augmentation
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

---

## 4. Tech Stack

### 4.1. Core ML/DL

| Thành phần   | Công nghệ            | Phiên bản | Lý do chọn                 |
| ------------ | -------------------- | --------- | -------------------------- |
| Framework    | **PyTorch**          | 2.1+      | Linh hoạt, ecosystem mạnh  |
| Vision       | **timm**             | 0.9+      | Pre-trained models đa dạng |
| Backbone     | **EfficientNet-B0**  | -         | Balance accuracy/speed     |
| XAI          | **pytorch-grad-cam** | 1.4+      | Grad-CAM, Grad-CAM++       |
| Optimization | **ONNX Runtime**     | 1.16+     | Inference optimization     |

### 4.2. Data Processing

| Thành phần       | Công nghệ                | Mục đích           |
| ---------------- | ------------------------ | ------------------ |
| Image Processing | **Pillow**, **OpenCV**   | Load, resize       |
| Augmentation     | **Albumentations**       | Data augmentation  |
| Dataset          | **HuggingFace Datasets** | Dataset management |

### 4.3. Training Infrastructure

| Thành phần          | Công nghệ                          | Mục đích               |
| ------------------- | ---------------------------------- | ---------------------- |
| Experiment Tracking | **Weights & Biases**               | Logging, visualization |
| Config              | **Hydra** / **yaml**               | Configuration          |
| GPU                 | **Google Colab Pro+** / **Kaggle** | Training               |

### 4.4. Web Application

| Layer         | Công nghệ        | Mục đích            |
| ------------- | ---------------- | ------------------- |
| Backend       | **FastAPI**      | REST API            |
| Frontend      | **Gradio**       | Rapid prototyping   |
| Model Serving | **ONNX Runtime** | Optimized inference |

### 4.5. Development Tools

| Công cụ          | Mục đích             |
| ---------------- | -------------------- |
| **uv** / **pip** | Package management   |
| **Ruff**         | Linting + Formatting |
| **pytest**       | Unit testing         |

---

## 5. Kiến trúc hệ thống

### 5.1. Model Architecture (Phased Approach)

#### Phase 1: Single-Branch Baseline (Bắt đầu với cái này)

```
┌─────────────────────────────────────────────────────────────┐
│                    BASELINE ARCHITECTURE                    │
│                    (EfficientNet-B0)                        │
└─────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────┐
        │              INPUT IMAGE                │
        │              (224 x 224 x 3)            │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │          PREPROCESSING                  │
        │   • Resize to 224x224                   │
        │   • Normalize (ImageNet stats)          │
        │   • JPEG augmentation (train only)      │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │         EFFICIENTNET-B0                 │
        │         (Pretrained ImageNet)           │
        │                                         │
        │   • Conv Stem                           │
        │   • MBConv Blocks (x16)                 │
        │   • Global Average Pooling              │
        │   • Feature: 1280-dim                   │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │         CLASSIFICATION HEAD             │
        │                                         │
        │   • Dropout(0.3)                        │
        │   • Linear(1280 → 1)                    │
        │   • Sigmoid                             │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │              OUTPUT                     │
        │    P(Fake) ∈ [0, 1]                     │
        │    + Grad-CAM Heatmap                   │
        └─────────────────────────────────────────┘
```

#### Phase 1.5: Optional Frequency Branch (Nếu còn thời gian)

```
┌─────────────────────────────────────────────────────────────┐
│              OPTIONAL: DUAL-BRANCH ARCHITECTURE             │
└─────────────────────────────────────────────────────────────┘

                    INPUT IMAGE (224 x 224 x 3)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌───────────────────┐           ┌───────────────────┐
    │  SPATIAL BRANCH   │           │ FREQUENCY BRANCH  │
    │  (EfficientNet)   │           │   (SRM + DCT)     │
    │                   │           │                   │
    │  Features: 1280   │           │  Features: 256    │
    └─────────┬─────────┘           └─────────┬─────────┘
              │                               │
              └───────────┬───────────────────┘
                          ▼
                ┌───────────────────┐
                │  CONCAT + FC      │
                │  (1536 → 512 → 1) │
                └─────────┬─────────┘
                          ▼
                      P(Fake)
```

### 5.2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WEB APPLICATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         FRONTEND (Gradio)                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │   │
│  │  │ Image Upload │  │  Result View │  │  Heatmap Visualization   │   │   │
│  │  │              │  │  (Real/Fake) │  │  (Grad-CAM overlay)      │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      BACKEND (FastAPI)                              │   │
│  │                                                                     │   │
│  │  POST /api/predict    → Inference + probability                     │   │
│  │  POST /api/explain    → Grad-CAM heatmap                           │   │
│  │  GET  /api/health     → Service health check                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       MODEL SERVICE                                 │   │
│  │  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────┐     │   │
│  │  │ Preprocessor│  │  ONNX Runtime   │  │   Grad-CAM Engine   │     │   │
│  │  │             │  │  (INT8 quant)   │  │                     │     │   │
│  │  └─────────────┘  └─────────────────┘  └─────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Roadmap theo Phase

### 6.1. Tổng quan Timeline

```
2025                                              2026
Nov          Dec          Jan          Feb          Mar          Apr          May
 |────────────|────────────|────────────|────────────|────────────|────────────|
 │◄────── PHASE 1 ───────►│◄────── PHASE 2 ────────►│◄─────── PHASE 3 ───────►│
 │   Foundation & Data    │  Model Dev & Optimize  │   Application & Report  │
 │                        │                        │                         │
 │ Sprint 1   │ Sprint 2  │ Sprint 3   │ Sprint 4  │ Sprint 5    │ Sprint 6  │
 │ Setup/Data │ Baseline  │ Fusion     │ XAI/Opt   │ Web Demo    │ Report    │
```

### 6.2. Chi tiết các Phase

---

## 📦 PHASE 1: Foundation & Data (T11-T12/2025)

> **Mục tiêu Phase**: Thiết lập nền tảng dự án và xây dựng bộ dữ liệu chuẩn
> **Thời gian**: 2 tháng
> **Hoàng**: Environment, Model, Training | **Luân**: Download data theo hướng dẫn

### Sprint 1.1: Project Setup & Data Collection (T11/2025)

**Mục tiêu Sprint**: Thiết lập môi trường và thu thập dữ liệu thô

| Task ID | Task                                    | Subtasks                                       | Assignee | Status |
| ------- | --------------------------------------- | ---------------------------------------------- | -------- | ------ |
| 1.1.1   | **Environment Setup**                   |                                                | Hoàng    | ⬜     |
|         |                                         | 1.1.1.1 Khởi tạo Git repository                |          | ⬜     |
|         |                                         | 1.1.1.2 Setup cấu trúc thư mục theo chuẩn      |          | ⬜     |
|         |                                         | 1.1.1.3 Cấu hình pyproject.toml + requirements |          | ⬜     |
|         |                                         | 1.1.1.4 Setup Weights & Biases project         |          | ⬜     |
|         |                                         | 1.1.1.5 Tạo Colab notebook template            |          | ⬜     |
| 1.1.2   | **Data Download - Real** ✨             |                                                | Luân     | ⬜     |
|         | _(Việc nhẹ - theo hướng dẫn của Hoàng)_ | 1.1.2.1 Download FFHQ dataset (theo script)    |          | ⬜     |
|         |                                         | 1.1.2.2 Download CelebA-HQ (theo link)         |          | ⬜     |
|         |                                         | 1.1.2.3 Sắp xếp vào đúng folder                |          | ⬜     |
| 1.1.3   | **Data Collection - Fake (GAN)**        |                                                | Hoàng    | ⬜     |
|         |                                         | 1.1.3.1 Download DFFD fake subset              |          | ⬜     |
|         |                                         | 1.1.3.2 Generate StyleGAN2 faces (10k)         |          | ⬜     |
|         |                                         | 1.1.3.3 Generate ProGAN faces (5k)             |          | ⬜     |
| 1.1.4   | **Data Collection - Fake (Diffusion)**  |                                                | Hoàng    | ⬜     |
|         |                                         | 1.1.4.1 Download GenImage subset (20k)         |          | ⬜     |
|         |                                         | 1.1.4.2 Generate SD v1.5 faces (10k)           |          | ⬜     |
|         |                                         | 1.1.4.3 Prepare OOD test set (SDXL, MJ proxy)  |          | ⬜     |

**✅ Milestone 1.1**: Raw dataset collected (≥40,000 ảnh)

---

### Sprint 1.2: Data Pipeline & Baseline Model (T12/2025)

**Mục tiêu Sprint**: Xử lý dữ liệu và train mô hình baseline

| Task ID | Task                    | Subtasks                                     | Assignee | Status |
| ------- | ----------------------- | -------------------------------------------- | -------- | ------ |
| 1.2.1   | **Data Pipeline**       |                                              | Hoàng    | ⬜     |
|         |                         | 1.2.1.1 Implement Dataset class (PyTorch)    |          | ⬜     |
|         |                         | 1.2.1.2 Implement augmentation pipeline      |          | ⬜     |
|         |                         | 1.2.1.3 Train/Val/Test-OOD split             |          | ⬜     |
|         |                         | 1.2.1.4 Create data manifest files           |          | ⬜     |
| 1.2.2   | **Model Architecture**  |                                              | Hoàng    | ⬜     |
|         |                         | 1.2.2.1 Implement EfficientNet-B0 classifier |          | ⬜     |
|         |                         | 1.2.2.2 Create model factory                 |          | ⬜     |
|         |                         | 1.2.2.3 Unit test model forward pass         |          | ⬜     |
| 1.2.3   | **Training Pipeline**   |                                              | Hoàng    | ⬜     |
|         |                         | 1.2.3.1 Implement Trainer class              |          | ⬜     |
|         |                         | 1.2.3.2 Setup BCE/Focal Loss                 |          | ⬜     |
|         |                         | 1.2.3.3 LR scheduler (CosineAnnealing)       |          | ⬜     |
|         |                         | 1.2.3.4 Early stopping callback              |          | ⬜     |
|         |                         | 1.2.3.5 Wandb logging integration            |          | ⬜     |
| 1.2.4   | **Baseline Training**   |                                              | Hoàng    | ⬜     |
|         |                         | 1.2.4.1 Train EfficientNet-B0 (RGB only)     |          | ⬜     |
|         |                         | 1.2.4.2 Hyperparameter tuning (LR, batch)    |          | ⬜     |
|         |                         | 1.2.4.3 Save best checkpoint                 |          | ⬜     |
| 1.2.5   | **Baseline Evaluation** |                                              | Hoàng    | ⬜     |
|         |                         | 1.2.5.1 Compute metrics (AUC, Acc, F1)       |          | ⬜     |
|         |                         | 1.2.5.2 Generate confusion matrix            |          | ⬜     |
|         |                         | 1.2.5.3 Plot ROC curve                       |          | ⬜     |
|         |                         | 1.2.5.4 Per-source accuracy breakdown        |          | ⬜     |

**✅ Milestone 1.2**: Dataset v1 (≥20,000 processed) + Baseline AUC ≥ 0.88 (in-domain)

**📊 Phase 1 Deliverables**:

- [ ] Dataset v1 với manifest files
- [ ] Baseline model checkpoint (.pt)
- [ ] Training logs trên W&B
- [ ] Báo cáo kết quả Phase 1

---

## 📦 PHASE 2: Model Development & Optimization (T01-T02/2026)

> **Mục tiêu Phase**: Phát triển mô hình Fusion và tích hợp XAI
> **Thời gian**: 2 tháng
> **Hoàng**: Toàn bộ model dev, XAI, optimization | **Luân**: Chạy test theo script

### Sprint 2.1: Fusion Model & Benchmark (T01/2026)

**Mục tiêu Sprint**: Xây dựng dual-branch architecture và benchmark với SOTA

| Task ID | Task                       | Subtasks                                       | Assignee | Status |
| ------- | -------------------------- | ---------------------------------------------- | -------- | ------ |
| 2.1.1   | **Frequency Branch**       |                                                | Hoàng    | ⬜     |
|         |                            | 2.1.1.1 Implement SRM filter module            |          | ⬜     |
|         |                            | 2.1.1.2 Implement DCT transform                |          | ⬜     |
|         |                            | 2.1.1.3 Build frequency feature extractor      |          | ⬜     |
|         |                            | 2.1.1.4 Unit test frequency module             |          | ⬜     |
| 2.1.2   | **Fusion Architecture**    |                                                | Hoàng    | ⬜     |
|         |                            | 2.1.2.1 Implement Attention-based Fusion       |          | ⬜     |
|         |                            | 2.1.2.2 Integrate spatial + frequency branches |          | ⬜     |
|         |                            | 2.1.2.3 Train fusion model                     |          | ⬜     |
|         |                            | 2.1.2.4 Compare with baseline (ablation)       |          | ⬜     |
| 2.1.3   | **Reproduce SOTA Methods** |                                                | Hoàng    | ⬜     |
|         |                            | 2.1.3.1 Setup CNNDetection (Wang et al.)       |          | ⬜     |
|         |                            | 2.1.3.2 Run pretrained on our test set         |          | ⬜     |
|         |                            | 2.1.3.3 Setup UniversalFakeDetect              |          | ⬜     |
|         |                            | 2.1.3.4 Run CLIP-based on our test set         |          | ⬜     |
| 2.1.4   | **OOD Evaluation**         |                                                | Hoàng    | ⬜     |
|         |                            | 2.1.4.1 Evaluate on SDXL (unseen)              |          | ⬜     |
|         |                            | 2.1.4.2 Evaluate on MJ proxy (unseen)          |          | ⬜     |
|         |                            | 2.1.4.3 Per-source breakdown analysis          |          | ⬜     |
|         |                            | 2.1.4.4 Create comparison table                |          | ⬜     |

**✅ Milestone 2.1**: Fusion model + Comparison report (≥3 methods)

---

### Sprint 2.2: XAI & Model Optimization (T02/2026)

**Mục tiêu Sprint**: Tích hợp Grad-CAM và tối ưu hóa model

| Task ID | Task                      | Subtasks                                         | Assignee | Status |
| ------- | ------------------------- | ------------------------------------------------ | -------- | ------ |
| 2.2.1   | **Grad-CAM Integration**  |                                                  | Hoàng    | ⬜     |
|         |                           | 2.2.1.1 Integrate pytorch-grad-cam               |          | ⬜     |
|         |                           | 2.2.1.2 Implement heatmap overlay function       |          | ⬜     |
|         |                           | 2.2.1.3 Generate XAI gallery (50 samples)        |          | ⬜     |
|         |                           | 2.2.1.4 Validate heatmap highlights face regions |          | ⬜     |
| 2.2.2   | **Robustness Testing**    |                                                  | Hoàng    | ⬜     |
|         |                           | 2.2.2.1 Test JPEG compression (q=60, 80)         |          | ⬜     |
|         |                           | 2.2.2.2 Test resize (0.5x, 0.75x, 1.5x)          |          | ⬜     |
|         |                           | 2.2.2.3 Test crop (center, random)               |          | ⬜     |
|         |                           | 2.2.2.4 Create robustness report                 |          | ⬜     |
| 2.2.3   | **Model Export**          |                                                  | Hoàng    | ⬜     |
|         |                           | 2.2.3.1 Export to ONNX format                    |          | ⬜     |
|         |                           | 2.2.3.2 Apply INT8 quantization                  |          | ⬜     |
|         |                           | 2.2.3.3 Validate ONNX output matches PyTorch     |          | ⬜     |
| 2.2.4   | **Speed Benchmark** ✨    |                                                  | Luân     | ⬜     |
|         | _(Chạy script của Hoàng)_ | 2.2.4.1 Chạy benchmark script                    |          | ⬜     |
|         |                           | 2.2.4.2 Ghi lại kết quả vào bảng                 |          | ⬜     |

**✅ Milestone 2.2**: Final model (AUC ≥ 0.92 in-domain, ≥ 0.85 OOD) + XAI + ONNX

**📊 Phase 2 Deliverables**:

- [ ] Fusion model checkpoint (.pt + .onnx)
- [ ] Comparison report với 3+ SOTA methods
- [ ] XAI gallery (50 heatmap samples)
- [ ] Robustness report
- [ ] Speed benchmark report

---

## 📦 PHASE 3: Application & Report (T03-T05/2026)

> **Mục tiêu Phase**: Xây dựng Web Demo và hoàn thiện báo cáo
> **Thời gian**: 3 tháng
> **Hoàng**: Backend, API, Integration | **Luân**: Test UI, Viết Chương 1-2 báo cáo

### Sprint 3.1: Web Demo Development (T03/2026)

**Mục tiêu Sprint**: Xây dựng ứng dụng web hoàn chỉnh

| Task ID | Task                        | Subtasks                                        | Assignee | Status |
| ------- | --------------------------- | ----------------------------------------------- | -------- | ------ |
| 3.1.1   | **Backend API**             |                                                 | Hoàng    | ⬜     |
|         |                             | 3.1.1.1 Setup FastAPI project                   |          | ⬜     |
|         |                             | 3.1.1.2 Implement POST /api/predict             |          | ⬜     |
|         |                             | 3.1.1.3 Implement POST /api/explain             |          | ⬜     |
|         |                             | 3.1.1.4 Implement GET /api/health               |          | ⬜     |
|         |                             | 3.1.1.5 Add request validation                  |          | ⬜     |
| 3.1.2   | **Model Service**           |                                                 | Hoàng    | ⬜     |
|         |                             | 3.1.2.1 Load ONNX model on startup              |          | ⬜     |
|         |                             | 3.1.2.2 Implement preprocessing pipeline        |          | ⬜     |
|         |                             | 3.1.2.3 Implement Grad-CAM service              |          | ⬜     |
|         |                             | 3.1.2.4 Add error handling                      |          | ⬜     |
| 3.1.3   | **Frontend UI**             |                                                 | Hoàng    | ⬜     |
|         |                             | 3.1.3.1 Setup Gradio interface                  |          | ⬜     |
|         |                             | 3.1.3.2 Image upload component                  |          | ⬜     |
|         |                             | 3.1.3.3 Result display (Real/Fake + confidence) |          | ⬜     |
|         |                             | 3.1.3.4 Heatmap visualization                   |          | ⬜     |
|         |                             | 3.1.3.5 UI styling và UX polish                 |          | ⬜     |
| 3.1.4   | **UI Testing** ✨           |                                                 | Luân     | ⬜     |
|         | _(Test và góp ý cho Hoàng)_ | 3.1.4.1 Test upload nhiều loại ảnh              |          | ⬜     |
|         |                             | 3.1.4.2 Ghi nhận lỗi và feedback                |          | ⬜     |
|         |                             | 3.1.4.3 Test trên nhiều thiết bị                |          | ⬜     |
| 3.1.5   | **Integration**             |                                                 | Hoàng    | ⬜     |
|         |                             | 3.1.5.1 End-to-end testing                      |          | ⬜     |
|         |                             | 3.1.5.2 Latency optimization (target ≤ 2s)      |          | ⬜     |
|         |                             | 3.1.5.3 Error case handling                     |          | ⬜     |
|         |                             | 3.1.5.4 Deploy to local/Colab                   |          | ⬜     |

**✅ Milestone 3.1**: Working web demo (latency ≤ 2s/ảnh)

---

### Sprint 3.2: Documentation & Defense Prep (T04-T05/2026)

**Mục tiêu Sprint**: Hoàn thiện tài liệu và chuẩn bị bảo vệ

| Task ID | Task                            | Subtasks                                    | Assignee | Status |
| ------- | ------------------------------- | ------------------------------------------- | -------- | ------ |
| 3.2.1   | **Báo cáo - Chương 1-2** ✨     |                                             | Luân     | ⬜     |
|         | _(Viết theo outline của Hoàng)_ | 3.2.1.1 Chương 1: Mở đầu (theo mẫu)         |          | ⬜     |
|         |                                 | 3.2.1.2 Chương 2: Tổng quan (theo tài liệu) |          | ⬜     |
|         |                                 | 3.2.1.3 Gửi Hoàng review                    |          | ⬜     |
| 3.2.2   | **Báo cáo - Chương 3-4-5**      |                                             | Hoàng    | ⬜     |
|         |                                 | 3.2.2.1 Chương 3: Phương pháp và Xây dựng   |          | ⬜     |
|         |                                 | 3.2.2.2 Chương 4: Kết quả thực nghiệm       |          | ⬜     |
|         |                                 | 3.2.2.3 Chương 5: Kết luận                  |          | ⬜     |
|         |                                 | 3.2.2.4 Tạo bảng, biểu đồ, hình ảnh         |          | ⬜     |
| 3.2.3   | **Tổng hợp báo cáo**            |                                             | Hoàng    | ⬜     |
|         |                                 | 3.2.3.1 Merge Chương 1-2 của Luân           |          | ⬜     |
|         |                                 | 3.2.3.2 Format theo mẫu trường              |          | ⬜     |
|         |                                 | 3.2.3.3 Review với GVHD                     |          | ⬜     |
| 3.2.4   | **Defense Preparation**         |                                             | Hoàng    | ⬜     |
|         |                                 | 3.2.4.1 Tạo slide thuyết trình              |          | ⬜     |
|         |                                 | 3.2.4.2 Quay video demo                     |          | ⬜     |
|         |                                 | 3.2.4.3 Chuẩn bị Q&A                        |          | ⬜     |
| 3.2.5   | **Hỗ trợ Defense** ✨           |                                             | Luân     | ⬜     |
|         | _(Hỗ trợ nhẹ)_                  | 3.2.5.1 Chuẩn bị ảnh test cho demo          |          | ⬜     |
|         |                                 | 3.2.5.2 Luyện tập thuyết trình cùng Hoàng   |          | ⬜     |
| 3.2.6   | **Final Deliverables**          |                                             | Hoàng    | ⬜     |
|         |                                 | 3.2.6.1 Đóng gói source code                |          | ⬜     |
|         |                                 | 3.2.6.2 Tạo README hướng dẫn                |          | ⬜     |
|         |                                 | 3.2.6.3 Export model weights                |          | ⬜     |
|         |                                 | 3.2.6.4 Chuẩn bị hồ sơ nghiệm thu           |          | ⬜     |

**✅ Milestone 3.2**: Hồ sơ nghiệm thu đầy đủ

**📊 Phase 3 Deliverables**:

- [ ] Web application hoạt động
- [ ] Báo cáo tổng kết (Docx + PDF)
- [ ] Slide thuyết trình
- [ ] Video demo
- [ ] Source code đóng gói + README
- [ ] Model weights (.pt + .onnx)

---

## 7. Tổng hợp Task Tracking

### 7.1. Task Summary by Phase

| Phase                | Sprints | Total Tasks | Total Subtasks |
| -------------------- | ------- | ----------- | -------------- |
| Phase 1: Foundation  | 2       | 10          | 28             |
| Phase 2: Model Dev   | 2       | 8           | 24             |
| Phase 3: Application | 2       | 9           | 27             |
| **Total**            | **6**   | **27**      | **79**         |

### 7.2. Task Status Legend

| Symbol | Meaning     |
| ------ | ----------- |
| ⬜     | Not Started |
| 🔄     | In Progress |
| ✅     | Completed   |
| ⏸️     | Blocked     |
| ❌     | Cancelled   |

### 7.3. Milestone Summary

| Milestone                | Phase   | Target Date | KPI                           | Status |
| ------------------------ | ------- | ----------- | ----------------------------- | ------ |
| M1.1: Raw Dataset        | Phase 1 | 30/11/2025  | ≥40k ảnh                      | ⬜     |
| M1.2: Baseline Model     | Phase 1 | 31/12/2025  | AUC ≥ 0.88                    | ⬜     |
| M2.1: Fusion + Benchmark | Phase 2 | 31/01/2026  | 3+ methods compared           | ⬜     |
| M2.2: Final Model + XAI  | Phase 2 | 28/02/2026  | AUC ≥ 0.92 (ID), ≥ 0.85 (OOD) | ⬜     |
| M3.1: Web Demo           | Phase 3 | 31/03/2026  | Latency ≤ 2s                  | ⬜     |
| M3.2: Defense Ready      | Phase 3 | 15/05/2026  | Full package                  | ⬜     |

### 7.4. Weekly Progress Template

```markdown
## Week X Progress (DD/MM/YYYY)

### Completed

- [ ] Task X.X.X.X: Description

### In Progress

- [ ] Task X.X.X.X: Description (XX% done)

### Blockers

- Issue: Description
- Action needed: ...

### Next Week Plan

- [ ] Task X.X.X.X: Description
```

---

## 8. KPIs & Metrics

### 8.1. Model Performance KPIs (Thực tế)

| Metric       | In-Domain | OOD    | Priority    |
| ------------ | --------- | ------ | ----------- |
| **AUC-ROC**  | ≥ 0.90    | ≥ 0.75 | 🔴 Critical |
| **Accuracy** | ≥ 88%     | ≥ 75%  | 🔴 Critical |
| **F1-Score** | ≥ 0.88    | ≥ 0.75 | 🟡 High     |

### 8.2. Robustness KPIs

| Condition    | Max AUC Drop |
| ------------ | ------------ |
| JPEG q=60    | ≤ 8%         |
| JPEG q=80    | ≤ 3%         |
| Resize 0.75x | ≤ 5%         |

### 8.3. System KPIs

| Metric        | Target     |
| ------------- | ---------- |
| Latency (CPU) | ≤ 2s/image |
| Model Size    | ≤ 50MB     |

### 8.4. Comparison KPIs

| Requirement      | Target |
| ---------------- | ------ |
| Methods compared | ≥ 3    |
| OOD evaluation   | Yes    |

---

## 9. Evaluation Protocol

### 9.1. Comparison Table Template

| Method              | Year | In-domain AUC | OOD AUC | Params |
| ------------------- | ---- | ------------- | ------- | ------ |
| Wang et al.         | 2020 | 0.99          | 0.78    | 25M    |
| UniversalFakeDetect | 2023 | 0.95          | 0.82    | 300M   |
| **Ours (Baseline)** | 2026 | ?             | ?       | 5M     |

### 9.2. Per-Source Breakdown

| Source    | Type          | AUC | Notes     |
| --------- | ------------- | --- | --------- |
| StyleGAN2 | GAN (seen)    | ?   | In-domain |
| SDXL      | Diff (unseen) | ?   | OOD       |

---

## 10. Cấu trúc thư mục

```
HolmHz/
├── data/                    # Dataset (gitignored)
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test_ood/
│   └── manifests/
├── src/                     # Source code
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── xai/
│   └── inference/
├── app/                     # Web application
├── configs/
├── notebooks/
├── scripts/
├── docs/
├── outputs/                 # Checkpoints (gitignored)
├── pyproject.toml
└── README.md
```

---

## 11. Hướng mở rộng

### 11.1. Phase 2: Video (Sau T05/2026)

```
Nếu Phase 1 hoàn thành sớm:
├── Frame extraction (1 FPS)
├── Frame-level → Video-level aggregation
├── Datasets: FaceForensics++, Celeb-DF
└── New KPIs: Video-level AUC
```

### 11.2. Phase 3: Audio (Tương lai)

```
├── Mel-spectrogram / MFCC
├── Models: RawNet2, AASIST
├── Datasets: ASVspoof, WaveFake
└── Applications: Voice verification
```

### 11.3. Roadmap dài hạn

```
2025-2026         2026-2027          2027+
─────────         ─────────          ─────
Phase 1: Image    Phase 2: Video     Phase 3: Multi-modal
• CNN baseline    • Temporal         • Audio + Video
• Benchmark       • FaceForensics++  • Real-time
• XAI             • Video demo       • Mobile
```

---

## ⚠️ Disclaimer

1. **Nghiên cứu ứng dụng** - Không claim novelty về kiến trúc mới
2. **Kết quả tham khảo** - Không thay thế giám định pháp lý
3. **OOD là open problem** - Cross-dataset generalization chưa giải quyết được
4. **Proof-of-concept** - Không phải sản phẩm thương mại

---

## ✅ Tiêu chí thành công

| Tiêu chí           | Mức đạt | Mức vượt |
| ------------------ | ------- | -------- |
| In-domain AUC      | ≥ 0.88  | ≥ 0.92   |
| OOD AUC            | ≥ 0.70  | ≥ 0.78   |
| Comparison methods | ≥ 2     | ≥ 4      |
| Web demo           | Working | + Docker |

---

**Last Updated:** 02/02/2026  
**Author:** Lê Văn Hoàng  
**Version:** 2.0 (Revised based on critical analysis)
