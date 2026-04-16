# Table 4.1 — Cross-Domain Evaluation of AI-Generated Image Detection Methods

> Dataset: HolmHz-v2 (Train: 28,220 | Test ID: 3,526 | Test OOD: 182)
> Platform: Kaggle T4 ×2, AdamW, CosineAnnealing, 30 epochs
> Bold = best per column. ↑ = higher is better. † = training diverged.

| Group | Method | Architecture | Params | ID AUC↑ | ID Acc↑ | ID F1↑ | OOD AUC↑ | OOD Acc↑ | OOD F1↑ |
|:------|:-------|:------------|:------:|:-------:|:-------:|:------:|:--------:|:--------:|:-------:|
| Baseline | CNNDetection | ResNet-50 (Wang 2020) | ~23M | 0.6619 | 0.5244 | 0.0368 | 0.3253 | 0.5165 | 0.0000 |
| Baseline | UnivFakeDetect | CLIP ViT-L/14 (Ojha 2023) | ~304M | 0.7218 | 0.7153 | 0.6270 | 0.4858 | 0.5330 | 0.0860 |
| Baseline | DeepfakeBench | EffNet-B4 (Yan 2023) | ~19M | 0.4389 | 0.4504 | 0.4055 | 0.5359 | 0.5385 | 0.4878 |
| Ours | EfficientNet-B0 | EfficientNet-B0 (v9) | 4M | 0.9984 | 0.9844 | 0.9839 | 0.8963 | 0.7802 | 0.7849 |
| Ours | ResNet-18 | ResNet-18 | 11M | 0.9953 | 0.9711 | 0.9702 | 0.8646 | 0.8022 | 0.7955 |
| Ours | ViT-Small/16 | ViT-Small/16 | 22M | 0.9741 | 0.9209 | 0.9201 | 0.8331 | 0.7473 | 0.7444 |
| Ours | Swin-Tiny† | Swin-T (failed) | 28M | 0.6195 | 0.5366 | 0.6331 | 0.8112 | 0.6758 | 0.7204 |

## Key Findings

1. **EfficientNet-B0 (Ours)** achieves highest ID AUC (0.9984) and OOD AUC (0.8963)
2. **ResNet-18 (Ours)** achieves highest OOD Accuracy (0.8022) with balanced precision/recall
3. All 3 research baselines perform near-random (AUC < 0.73) — trained on GAN/video data, fail on Diffusion
4. Swin-Tiny diverged during training — larger models need specialized hyperparameters on small datasets
5. **"Ours" models dramatically outperform SOTA baselines** on both ID and OOD evaluation
