# HolmHz 🕵️‍♂️

<div align="center">

**AI-Generated Image Detection System**  
*"You see, but you do not observe." — Sherlock Holmes*

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

**HolmHz** is a research system for detecting AI-generated (synthetic) images using Convolutional Neural Networks with Transfer Learning. It trains and benchmarks four modern CNN/Transformer architectures against three international SOTA baselines on a custom curated dataset covering both GAN and Diffusion-era generators.

> **Name origin**: *Holmes* (deduction) + *Hz* (frequency) — the system detects invisible frequency artifacts left by generative models.

### Key Results

| Model | Params | ID AUC ↑ | ID Acc ↑ | OOD AUC ↑ | OOD Acc ↑ |
|:------|:------:|:--------:|:--------:|:---------:|:---------:|
| **EfficientNet-B0 v9 (Ours)** | **4M** | **0.998** | **98.4%** | **0.896** | 78.0% |
| ResNet-18 (Ours) | 11M | 0.995 | 97.1% | 0.865 | **80.2%** |
| ViT-Small/16 (Ours) | 22M | 0.974 | 92.1% | 0.833 | 74.7% |
| CNNDetection (Wang 2020) | ~23M | 0.662 | 52.4% | 0.325 | 51.7% |
| UniversalFakeDetect (Ojha 2023) | ~304M | 0.722 | 71.5% | 0.486 | 53.3% |
| DeepfakeBench (Yan 2023) | ~19M | 0.439 | 45.0% | 0.536 | 53.9% |

> All models evaluated on the **same dataset** (Test ID: 3,526 images | Test OOD: 182 images) for fair comparison.

**Key finding**: EfficientNet-B0 (4M params) trained with JPEG Augmentation v3 outperforms a 304M-parameter SOTA model by **+0.41 OOD AUC** — demonstrating that data strategy matters more than model size.

---

## Features

- **Multi-architecture support**: EfficientNet-B0, ResNet-18, ViT-Small/16, Swin-Tiny via a unified `TimmDetector` wrapper
- **Registry pattern**: Config-driven model instantiation — swap architectures by changing one YAML field
- **JPEG Augmentation**: Critical technique that improves OOD generalization by 103.6% (AUC: 0.440 → 0.896)
- **Fair benchmarking**: HolmHz models vs 3 international SOTA baselines on identical data splits
- **Explainable AI**: Grad-CAM heatmap visualization integrated into the web demo
- **ONNX export**: Optimized inference ~1.5s/image on CPU (no GPU required for demo)
- **Weighted sampling**: Handles class and source imbalance across 5 dataset sources

---

## Dataset

**HolmHz-v2** — 35,454 images from 5 public Kaggle sources:

| Source | Content | Generator Type | Train Size |
|--------|---------|---------------|:----------:|
| RVF10K | Face images (CelebA + StyleGAN) | StyleGAN | 8,000 |
| DeepDetect-2025 | Diverse scenes | Diffusion mixed | 8,000 |
| Diffusion Fakes | DALL-E, Midjourney, SD, DeepFaceLab | 6+ generators | 4,024 |
| CIPLab Faces | Face manipulation (Chung-Ang Univ.) | Face manipulation | 3,266 |
| Camera vs AI | Real camera photos vs AI-generated | Mixed AI | 400 |

**Split strategy**:
- Train: 28,220 | Val: 3,526 | Test ID: 3,526 | Test OOD: 182
- OOD test set uses exclusively the Camera vs AI source — **not seen during training**

---

## Project Structure

```
HolmHz/
├── configs/                  # YAML training configs per model version
│   ├── train_v9.yaml         # EfficientNet-B0 v9 (best model)
│   ├── train_resnet18_v2.yaml
│   ├── train_vit_small_v2.yaml
│   └── train_swin_tiny_v2.yaml
├── src/holmhz/               # Core library
│   ├── backbones/            # Feature extractors (EfficientNet, Timm)
│   ├── detectors/            # Backbone + Classification head
│   ├── data/                 # Dataset, DataLoader, Augmentation
│   ├── training/             # Trainer, Loss, Scheduler
│   ├── evaluation/           # Evaluator, Metrics
│   ├── xai/                  # Grad-CAM explainer
│   └── utils/                # Registry pattern, Logger
├── scripts/                  # Training, evaluation, benchmark scripts
├── web/                      # Gradio demo app
│   ├── app.py
│   └── config.py
├── configs/                  # YAML training configs
├── outputs/
│   └── benchmark/            # Benchmark results and charts
│       ├── v2_benchmark_results.json
│       └── final_benchmark/  # Charts: AUC, Radar, Heatmap
├── docs/
│   ├── final_report.md       # Full academic report (Vietnamese)
│   └── REPORT_SOURCES.md     # All source files used in report
└── tests/                    # Unit tests
```

---

## Quick Start

### Prerequisites

```bash
python >= 3.9
pip install -r requirements.txt
```

### Run Web Demo

```bash
# Demo uses ResNet-18 ONNX (fastest, ~1.5s/image on CPU)
python web/app.py
```

Upload any image and receive:
- **Real/Fake classification** with confidence score
- **Grad-CAM heatmap** highlighting suspicious regions

### Training

```bash
# Train EfficientNet-B0 v9 (best model)
python scripts/train.py --config configs/train_v9.yaml

# Train all architectures
python scripts/train.py --config configs/train_resnet18_v2.yaml
python scripts/train.py --config configs/train_vit_small_v2.yaml
python scripts/train.py --config configs/train_swin_tiny_v2.yaml
```

### Evaluation & Benchmark

```bash
# Evaluate a single checkpoint
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_v9.pt --config configs/train_v9.yaml

# Run full 7-model benchmark
python scripts/benchmark.py
```

---

## Training Configuration (Best Model)

`configs/train_v9.yaml`:

```yaml
model:
  name: efficientnet_b0
  pretrained: true
  dropout: 0.3

training:
  epochs: 30
  batch_size: 32
  learning_rate: 0.0003      # AdamW
  weight_decay: 0.01
  scheduler: cosine           # Cosine Annealing
  pos_weight: 1.0
  early_stopping:
    patience: 7
    monitor: val_auc

data:
  image_size: 224
  augmentation: true          # JPEG quality 50-95 + Blur + Flip
  use_weighted_sampler: true  # Handles source imbalance
```

**Platform**: Kaggle Notebooks, NVIDIA Tesla T4 × 2 (DataParallel)

---

## Benchmark Results

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
Backbone (pretrained ImageNet)     → [B, feature_dim]
    │  EfficientNet-B0 → 1,280
    │  ResNet-18       → 512
    │  ViT-Small/16    → 384
    │  Swin-Tiny       → 768
    ▼
Dropout(p=0.3)
    ▼
Linear(feature_dim → 1)            → logits [B, 1]
    ▼
BCEWithLogitsLoss (training)
Sigmoid → P(Fake) ∈ [0,1] (inference)
```

Models are registered via `DETECTOR_REGISTRY` and instantiated config-driven — no code changes needed to swap architectures.

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

| Layer | Technology |
|-------|-----------|
| **ML Framework** | PyTorch 2.x, timm, Albumentations |
| **Inference** | ONNX Runtime (CPU-optimized) |
| **XAI** | pytorch-grad-cam |
| **Demo** | Gradio |
| **Experiment Tracking** | Weights & Biases (W&B) |
| **Testing** | pytest |
| **Linting** | Ruff |
| **IaC (proposed)** | Terraform |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker + AWS ECR |

---

## Contributions

**Lê Văn Hoàng** (Team Lead) — MSSV: 2224802010279 — D22CNTT02
- System architecture design and implementation (`src/holmhz/`)
- Dataset curation, manifest generation, and split strategy
- Model training (all 4 architectures), hyperparameter tuning
- Benchmark framework and SOTA comparison
- Web demo development and ONNX export
- Academic report writing

**Ngô Huỳnh Bảo Luân** — MSSV: 2524802010327 — D25CNTT10
- Data collection and preprocessing pipeline
- Evaluation scripts and metric computation
- Documentation support

**Supervisor**: ThS. Nguyễn Trung Kiệt  
**Institution**: Institute of Digital Technology, Thu Dau Mot University  
**Academic Year**: 2025–2026

---

## Citation

If you use HolmHz dataset, benchmark results, or methodology in your research, please cite:

```bibtex
@misc{holmhz2026,
  title     = {HolmHz: AI-Generated Image Detection with CNN and Transfer Learning},
  author    = {Lê Văn Hoàng and Ngô Huỳnh Bảo Luân},
  year      = {2026},
  note      = {NCKH SV Undergraduate Research, Thu Dau Mot University},
  url       = {https://github.com/EurusDevSec/HolmHz}
}
```

---

## References

1. Rössler et al. (2019) — FaceForensics++, ICCV
2. Wang et al. (2020) — CNNDetection, CVPR
3. Ojha et al. (2023) — UniversalFakeDetect, CVPR
4. Yan et al. (2023) — DeepfakeBench, arXiv
5. Tan & Le (2019) — EfficientNet, ICML
6. Selvaraju et al. (2017) — Grad-CAM, ICCV

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Made with ❤️ at <a href="https://tdmu.edu.vn">Thu Dau Mot University</a> · Institute of Digital Technology</sub>
</div>
