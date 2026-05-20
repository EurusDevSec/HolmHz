# HolmHz 🕵️‍♂️

<div align="center">

**AI-Generated Image Detection System**  
_"You see, but you do not observe." — Sherlock Holmes_

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4%20×2-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com)
[![EfficientNet-B0](https://img.shields.io/badge/Best%20Model-EfficientNet--B0-orange?style=flat-square)](configs/train_v9.yaml)
[![ID AUC](https://img.shields.io/badge/ID%20AUC-0.998-brightgreen?style=flat-square)]()
[![OOD AUC](https://img.shields.io/badge/OOD%20AUC-0.896-brightgreen?style=flat-square)]()

[📄 Report](docs/final_report.md) · [📊 Benchmark](outputs/benchmark/final_benchmark/benchmark_table_final.md) · [🏗️ AWS Architecture](docs/AWS_ARCHITECTURE_REVIEW.md)

</div>

---

## Overview

**HolmHz** is a research system for detecting AI-generated (synthetic) images using lightweight CNN models with Transfer Learning. It trains and benchmarks four modern architectures (EfficientNet-B0, ResNet-18, ViT-Small, Swin-Tiny) on a custom curated dataset and compares against three international SOTA baselines.

> **Name origin**: _Holmes_ (keen observation) + _Hz_ (frequency artifacts) — the system detects invisible frequency anomalies left by generative models.

**Use case**: Verify whether portrait/product images are AI-generated (Diffusion, DALL-E, Midjourney, StyleGAN) or authentic photos.

**Key achievement**: EfficientNet-B0 (4M params) achieves **0.896 OOD AUC** — outperforming a 304M-parameter SOTA model by **+0.41 AUC** — proving that data strategy (JPEG augmentation + weighted sampling) matters more than raw parameter count.

---

### Performance Comparison

| Model                           | Params | ID AUC ↑ | ID Acc ↑ | OOD AUC ↑ | OOD Acc ↑ |
| :------------------------------ | :----: | :------: | :------: | :-------: | :-------: |
| **EfficientNet-B0 v9** ⭐       |   4M   |  0.998   |  98.4%   |   0.896   |   78.0%   |
| ResNet-18 v2                    |  11M   |  0.995   |  97.1%   |   0.865   | **80.2%** |
| ViT-Small/16 v2                 |  22M   |  0.974   |  92.1%   |   0.833   |   74.7%   |
| Swin-Tiny v2                    |  28M   |  0.954   |  88.2%   |   0.802   |   71.3%   |
| CNNDetection (Wang 2020)        |  ~23M  |  0.662   |  52.4%   |   0.325   |   51.7%   |
| UniversalFakeDetect (Ojha 2023) |  304M  |  0.722   |  71.5%   |   0.486   |   53.3%   |
| DeepfakeBench (Yan 2023)        |  19M   |  0.439   |  45.0%   |   0.536   |   53.9%   |

**Legend**:

- ⭐ Best overall (highest OOD AUC)
- ID = In-distribution (test set from same sources as training)
- OOD = Out-of-distribution (test set from "Camera vs AI" source, never seen during training)
- All models tested on identical 3,708-image test set for fair comparison

---

## Features

## Features

- 🎯 **Multi-architecture support**: EfficientNet-B0, ResNet-18, ViT-Small/16, Swin-Tiny via unified `TimmDetector` wrapper
- ⚙️ **Config-driven design**: Swap models/hyperparams via YAML — no code changes needed
- 📈 **JPEG Augmentation v3**: Simulates camera compression artifacts, boosts OOD AUC by **+103.6%** (key insight)
- ⚖️ **Weighted sampling**: Handles class imbalance (real/fake) and source imbalance (5 dataset sources)
- 🏆 **Fair SOTA benchmark**: HolmHz models vs 3 published baselines on identical data splits
- 🔍 **Explainable AI**: Grad-CAM visualization shows suspicious regions (eyes, mouth, skin texture)
- ⚡ **ONNX export**: ~1.5s inference on CPU (ResNet-18), no GPU needed for inference
- 📊 **Experiment tracking**: Weights & Biases integration (optional, no signup required for local training)
- ✅ **Fully tested**: Unit tests for data pipeline, training loop, metrics, evaluation

---

## Dataset

**HolmHz-v2** — 35,454 images from 5 Kaggle datasets (split: Train/Val/Test ID/Test OOD)

### Dataset Composition

| Source              | Content                             | Generator       | # Images |    Split     |
| ------------------- | ----------------------------------- | --------------- | :------: | :----------: |
| **RVF10K**          | Faces (CelebA + StyleGAN)           | StyleGAN        |  8,000   |    Train     |
| **DeepDetect-2025** | Diverse scenes                      | Diffusion mixed |  8,000   |    Train     |
| **Diffusion Fakes** | DALL-E, Midjourney, SD, DeepFaceLab | 6+ generators   |  4,024   |    Train     |
| **CIPLab Faces**    | Face manipulation                   | Face-swap       |  3,266   |    Train     |
| **Camera vs AI**    | Real photos + AI-generated          | Mixed           |   400    | **OOD Test** |

**Split Strategy**:

- Train: 28,220 images
- Validation (ID): 3,526 images
- Test (ID): 3,526 images
- Test (OOD): 182 images ← **Never seen during training** (purity check)

### Download Dataset

```bash
# Automatically downloads on first training run
python scripts/train.py --config configs/train_v9.yaml
# Or manual download from Kaggle
kaggle datasets download -d [source-id]
# Extract to: data/raw_v2/
```

**Dataset stats**: [data/manifests_v2/](data/manifests_v2/) contains metadata (source, label, split assignments)

---

## Project Structure

```
HolmHz/
├── src/holmhz/                 # Core library (pip install -e .)
│   ├── backbones/              # CNN/Transformer backbones via timm
│   ├── detectors/              # Backbone + Classification head + Registry
│   ├── data/                   # Dataset, DataLoader, JPEG Augmentation
│   ├── training/               # Trainer, Loss functions, LR Scheduler
│   ├── evaluation/             # Evaluator, Metrics (AUC, Accuracy, F1)
│   ├── xai/                    # Grad-CAM explainability
│   └── utils/                  # Logger, Registry pattern, Helpers
│
├── scripts/                    # Standalone CLI tools
│   ├── train.py                # Train any architecture (Hydra config)
│   ├── evaluate.py             # Evaluate checkpoint on test sets
│   ├── benchmark.py            # 7-model SOTA comparison
│   ├── predict.py              # Single image inference
│   └── export_onnx.py          # Export to ONNX for prod
│
├── app/                        # Web interface
│   ├── gradio_ui.py            # Main Gradio demo
│   ├── api.py                  # FastAPI endpoints (optional)
│   └── schemas.py              # Request/response models
│
├── configs/                    # YAML training configs
│   ├── train_v9.yaml           # ✅ Best model (EfficientNet-B0)
│   ├── train_resnet18_v2.yaml
│   ├── train_vit_small_v2.yaml
│   └── train_swin_tiny_v2.yaml
│
├── outputs/                    # Generated artifacts
│   ├── checkpoints/            # Trained .pt files
│   ├── exports/                # ONNX models (.onnx)
│   ├── logs/                   # Training curves
│   └── benchmark/              # Comparison results, charts
│
├── data/                       # Dataset (auto-downloaded)
│   ├── raw_v2/                 # Raw images from Kaggle
│   ├── processed/              # Preprocessed/resized (optional)
│   └── manifests_v2/           # Metadata: train/val/test splits
│
├── docs/                       # Documentation
│   ├── final_report.md         # Full academic report (Vietnamese)
│   ├── CONTEXT.md              # Session notes for developers
│   └── plan.md                 # Project milestones
│
├── tests/                      # Unit tests (pytest)
│   ├── test_training.py
│   ├── test_data.py
│   └── test_detectors.py
│
├── pyproject.toml              # Package config (hatch/setuptools)
├── requirements.txt            # Runtime deps
├── requirements-dev.txt        # Dev deps (pytest, ruff, etc)
└── Makefile                    # Common commands (make train, make demo, etc)
```

**Key files for getting started**:

- 🚀 **[scripts/train.py](scripts/train.py)** — Entry point for training
- 📊 **[scripts/benchmark.py](scripts/benchmark.py)** — Reproduce benchmark results
- 🎨 **[app/gradio_ui.py](app/gradio_ui.py)** — Web demo
- 📋 **[configs/train_v9.yaml](configs/train_v9.yaml)** — Best model config

---

## Quick Start

### Prerequisites

```bash
# Minimum: Python 3.9+, GPU optional (demo works on CPU)
git clone https://github.com/EurusDevSec/HolmHz.git
cd HolmHz
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1️⃣ Run Web Demo (5 seconds)

```bash
# Downloads pretrained ResNet-18 ONNX model automatically
python app/gradio_ui.py
```

**What you get**:

- 🖼️ Image upload interface
- 📊 Real/Fake classification with confidence (0–100%)
- 🔍 Grad-CAM heatmap showing suspicious regions (red = likely manipulated)
- ⚡ ~1.5s inference on CPU, works on laptops

### 2️⃣ Model Selection Guide

| Use Case               | Model              | Pros                                         | Cons                           |
| ---------------------- | ------------------ | -------------------------------------------- | ------------------------------ |
| **Web Demo (Fast)**    | ResNet-18 ONNX     | Fastest (~1.5s/img), ~11M params             | Slightly lower OOD AUC (0.865) |
| **Best Accuracy**      | EfficientNet-B0 v9 | Highest AUC (0.998 ID, 0.896 OOD), 4M params | Needs PyTorch                  |
| **Research/Fine-tune** | ViT-Small v2       | Vision Transformer, better context awareness | 22M params, slower             |
| **Lightweight Edge**   | EfficientNet-B0    | Balances speed & accuracy                    | Needs GPU training             |

### 3️⃣ Training Your Own Model

```bash
# Train EfficientNet-B0 v9 (best results - recommended)
python scripts/train.py --config configs/train_v9.yaml

# Or train other architectures
python scripts/train.py --config configs/train_resnet18_v2.yaml
python scripts/train.py --config configs/train_vit_small_v2.yaml
```

**Outputs**:

- Checkpoint saved to `outputs/checkpoints/`
- Logs to Weights & Biases (optional, set `WANDB_API_KEY` in `.env`)

### 4️⃣ Evaluate & Benchmark

```bash
# Evaluate single model
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_v9.pt

# Run full 7-model SOTA benchmark (EfficientNet-B0 + 6 baselines)
python scripts/benchmark.py --output outputs/benchmark/results.json
```

**Benchmark results** including ID/OOD AUC, Accuracy, F1 saved to `outputs/benchmark/`

---

## Training Configuration (Best Model)

**Best model**: EfficientNet-B0 v9 ([configs/train_v9.yaml](configs/train_v9.yaml))

```yaml
# Model architecture
model:
  name: efficientnet_b0
  pretrained: true # ImageNet weights
  dropout: 0.3 # Prevents overfitting

# Training hyperparameters
training:
  epochs: 30
  batch_size: 32
  learning_rate: 0.0003 # AdamW optimizer
  weight_decay: 0.01 # L2 regularization
  scheduler: cosine # Cosine annealing LR decay
  pos_weight: 1.0 # Real/Fake class balance

  # Early stopping
  early_stopping:
    patience: 7
    monitor: val_auc

# Data augmentation (critical for generalization)
data:
  image_size: 224
  augmentation: true # JPEG quality 50-95 + Blur + Flip + Rotate
  use_weighted_sampler: true # Balances source imbalance
```

**Training environment**:

- Platform: Kaggle Notebooks T4 × 2 GPU (DataParallel)
- Time: ~45 min for 30 epochs
- No GPU needed for inference (ONNX export)

**Why this config wins**:

1. **JPEG augmentation** — Simulates real-world photo compression (phones, messengers, Instagram)
2. **Cosine annealing** — Smooth LR decay prevents abrupt drops
3. **Weighted sampler** — Prevents model from learning dataset artifacts (GAN-heavy sources)
4. **Early stopping** — Stops when validation AUC plateaus, prevents overfitting

---

## 🔬 Why EfficientNet-B0 Wins

**Key insight**: Parameter efficiency + data strategy > brute-force scaling

| Factor                | Impact                                                                   |
| --------------------- | ------------------------------------------------------------------------ |
| **JPEG Augmentation** | Simulates camera compression artifacts ⟹ +103.6% OOD AUC (0.440 → 0.896) |
| **Small model size**  | 4M params ⟹ less memorization, better generalization                     |
| **Efficient scaling** | EfficientNet scales depth/width/resolution proportionally                |
| **Weighted sampling** | Balances class & source imbalance ⟹ prevents GAN-bias                    |

**Result**: EfficientNet-B0 (4M) beats UniversalFakeDetect (304M) by **+0.41 OOD AUC** — demonstrating dataset and augmentation strategy > raw parameter count.

---

![ID vs OOD AUC Comparison](outputs/benchmark/final_benchmark/id_vs_ood_auc.png)

![Radar Chart — Multi-metric Comparison](outputs/benchmark/final_benchmark/radar_comparison.png)

![OOD Accuracy Heatmap per Source](outputs/benchmark/final_benchmark/ood_heatmap.png)

Full benchmark table: [`outputs/benchmark/final_benchmark/benchmark_table_final.md`](outputs/benchmark/final_benchmark/benchmark_table_final.md)

---

## Architecture

```
Input Image [B, 3, 224×224]
    │
    ▼
Backbone (ImageNet Pretrained)
    ├─ EfficientNet-B0  → [B, 1,280]
    ├─ ResNet-18        → [B, 512]
    ├─ ViT-Small/16     → [B, 384]
    └─ Swin-Tiny        → [B, 768]
    │
    ▼
Dropout(p=0.3)                   ← Regularization
    │
    ▼
Linear(feature_dim → 1)          → Logits [B, 1]
    │
    ▼
Training: BCEWithLogitsLoss
Inference: Sigmoid(logits) → P(Fake) ∈ [0, 1]
```

**Design principles**:

- 🔄 **Config-driven**: Change model via YAML, no code changes (Registry pattern)
- 🎯 **Transfer learning**: Leverage ImageNet pretrain for better generalization
- 📊 **Dropout + weight decay**: Combat overfitting on synthetic data
- ⚖️ **BCEWithLogitsLoss + pos_weight**: Handle real/fake class imbalance

---

## Proposed Cloud Deployment (AWS)

```
[User] → [Route 53] → [CloudFront + WAF]
                              │
              ┌───────────────┴────────────────┐
              │         AWS Region              │
              │  [API Gateway] → [Lambda]       │
              │       ↑              │          │
              │  [ECR] ←── GitHub    ├──→ [S3]  │
              │  [Secrets/SSM Mgr]  │           │
              │  [CloudWatch] ←─────┘           │
              └─────────────────────────────────┘
```

- **Compute**: Lambda Container Image (ONNX Runtime, ~1.5s/image)
- **Storage**: S3 for heatmap results, CloudFront CDN for delivery
- **IaC**: Terraform + GitHub Actions CI/CD pipeline
- **Cost**: ~$5/month at research-demo traffic

See [`docs/AWS_ARCHITECTURE_REVIEW.md`](docs/AWS_ARCHITECTURE_REVIEW.md) for full details.

---

## Tech Stack

| Layer                     | Technology                                            |
| ------------------------- | ----------------------------------------------------- |
| **ML Framework**          | PyTorch 2.x, timm (model zoo)                         |
| **Data**                  | Albumentations (augmentation), OpenCV (preprocessing) |
| **Inference**             | ONNX Runtime (CPU-optimized)                          |
| **Explainability**        | pytorch-grad-cam (Grad-CAM heatmaps)                  |
| **Web**                   | Gradio (no-code UI), FastAPI (optional)               |
| **Experiment Tracking**   | Weights & Biases (W&B) — optional, no signup required |
| **Configuration**         | Hydra + OmegaConf (YAML configs)                      |
| **Testing**               | pytest + pytest-cov                                   |
| **Code Quality**          | Ruff (linting/formatting)                             |
| **Deployment (proposed)** | Docker + AWS Lambda + S3                              |

---

## ❓ FAQ & Troubleshooting

### Q1: Can I run the demo on CPU?

**A**: Yes! Demo uses ONNX-optimized ResNet-18 (~1.5s/image on CPU). GPU optional.

### Q2: I don't have a Kaggle account. How do I get the dataset?

**A**:

- Option 1: Create free Kaggle account → install `kaggle` CLI → `kaggle datasets download -d [id]`
- Option 2: Download manually from Kaggle website
- Option 3: Script auto-downloads on first `python scripts/train.py` (with credentials)

### Q3: Which model should I use for production?

**A**:

- **Speed-critical (web)**: ResNet-18 ONNX (~1.5s/img)
- **Accuracy-critical**: EfficientNet-B0 v9 (~2s/img, +3.1% OOD AUC vs ResNet-18)

### Q4: Can I finetune on my own dataset?

**A**: Yes!

```bash
# 1. Update data/manifests_v2/train_manifest.json with your paths
# 2. Modify configs/train_v9.yaml → data.image_dir, data.label_column
# 3. python scripts/train.py --config configs/train_v9.yaml
```

### Q5: My GPU is out of memory (OOM). What do I do?

**A**:

- Reduce batch size: `train_v9.yaml` → `training.batch_size: 16` (default: 32)
- Reduce image size: `training.image_size: 196` (default: 224)
- Use gradient accumulation: `training.accumulation_steps: 2`

### Q6: How do I use the results in my research paper?

**A**: See **Citation** section. Benchmark results in [`outputs/benchmark/final_benchmark/`](outputs/benchmark/final_benchmark/).

### Q7: Can I export to TensorFlow/ONNX/CoreML?

**A**:

- ✅ ONNX: `python scripts/export_onnx.py --checkpoint outputs/checkpoints/best_v9.pt`
- ⚠️ TensorFlow: Not tested (PyTorch export only)
- ⚠️ CoreML: Manual conversion needed

### Q8: The model predicted wrong on my image. What's happening?

**A**:

1. Check Grad-CAM heatmap — is it focusing on relevant features?
2. If image is JPEG-compressed: lower OOD AUC (expected)
3. If image is from new generator: likely out-of-distribution (model was trained on GAN + Diffusion only)

### Q9: How do I set up Weights & Biases (W&B) logging?

**A**:

```bash
# 1. Create free account: https://wandb.ai
# 2. Get API key from account settings
# 3. Set env var: export WANDB_API_KEY=your-key-here
# 4. Training auto-logs to W&B
```

(Optional — training works without W&B)

### Q10: How do I reproduce the benchmark results?

**A**:

```bash
python scripts/benchmark.py --output outputs/benchmark/results.json
# Generates: AUC/Acc comparison, radar chart, heatmap
```

Takes ~30 min on GPU (all 7 models × all test sets).

---

## Contributions

This is an academic research project by two undergraduate students at Thu Dau Mot University (Vietnam).

| Name                    | MSSV          | Role                                                                       |
| ----------------------- | ------------- | -------------------------------------------------------------------------- |
| **Lê Văn Hoàng** (Lead) | 2224802010279 | System architecture, model training, benchmark framework, web demo, report |
| **Ngô Huỳnh Bảo Luân**  | 2524802010327 | Data collection, evaluation scripts, documentation support                 |

**Supervisor**: ThS. Nguyễn Trung Kiệt  
**Institution**: Institute of Digital Technology, Thu Dau Mot University, Ho Chi Minh City  
**Academic Year**: 2025–2026  
**Duration**: 7 months (Nov 2025 – May 2026)

---

## Citation

If you use HolmHz in your research, please cite:

```bibtex
@misc{holmhz2026,
  title     = {HolmHz: AI-Generated Image Detection with CNN Transfer Learning},
  author    = {Lê Văn Hoàng and Ngô Huỳnh Bảo Luân},
  year      = {2026},
  school    = {Institute of Digital Technology, Thu Dau Mot University},
  note      = {Undergraduate Research Project (NCKH SV)},
  url       = {https://github.com/EurusDevSec/HolmHz}
}
```

**Data Attribution**: Dataset sources from Kaggle:

1. RVF10K (StyleGAN faces)
2. DeepDetect-2025 (Diffusion mixed)
3. Diffusion Fakes (DALL-E, Midjourney, Stable Diffusion)
4. CIPLab Faces (Face manipulation)
5. Camera vs AI (Real + AI photos)

---

## References

**Foundational Papers**:

1. Rössler et al. (2019) — FaceForensics++, ICCV
2. Wang et al. (2020) — CNNDetection, CVPR
3. Tan & Le (2019) — EfficientNet: Rethinking Model Scaling for CNN, ICML
4. Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, ICCV

**SOTA Baselines**: 5. Ojha et al. (2023) — UniversalFakeDetect, CVPR 6. Yan et al. (2023) — DeepfakeBench, arXiv 7. Dosovitskiy & Beyer (2020) — An Image is Worth 16×16 Words: Transformers for Image Recognition, ICLR

**Related Work**:

- Li et al. (2018) — Capsule-Forensics: Using Capsule Networks to Detect Forged Images and Videos
- Frank et al. (2020) — Leveraging Frequency Analysis for Deep Fake Image Recognition
- Wodajo & Atnafu (2021) — Deepfake Video Detection through Optical Flow based CNN

---

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.

**Attribution**: When using this project, please cite the academic paper and acknowledge the authors (see Citation section).

---

## Contact & Support

- 📧 **Issues/Questions**: Open a GitHub issue or contact lê văn hoàng at 2224802010279@student.tdmu.edu.vn
- 📚 **Documentation**: [docs/final_report.md](docs/final_report.md) (full academic report in Vietnamese)
- 🏫 **Institution**: [Thu Dau Mot University](https://tdmu.edu.vn), Institute of Digital Technology

---

<div align="center">

**Made with ❤️ by the HolmHz research team at Thu Dau Mot University**

_"You see, but you do not observe." — Sherlock Holmes_

⭐ Star us on GitHub if this project was useful!

</div>
