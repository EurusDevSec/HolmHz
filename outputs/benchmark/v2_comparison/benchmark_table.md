# HolmHz v2 — Multi-Architecture Benchmark Results

> Generated: 2026-04-09 12:24
> Dataset: data/raw_v2 + data/manifests_v2 (Train: 28,220, Test ID: 3,526, Test OOD: 182)
> Platform: Kaggle T4 x2 (DataParallel)

## Table 4.1 — Overall Model Comparison

| Model | Type | Params | Best Epoch | Val AUC | ID AUC | ID Acc | ID F1 | OOD AUC | OOD Acc | OOD F1 |
|-------|------|--------|------------|---------|--------|--------|-------|---------|---------|--------|
| ResNet-18 | CNN | 11M | 28 | 0.9956 | 0.9953 | 0.9711 | 0.9702 | 0.8646 | 0.8022 | 0.7955 |
| ViT-Small/16 | Vision Transformer | 22M | 29 | 0.9735 | 0.9741 | 0.9209 | 0.9201 | 0.8331 | 0.7473 | 0.7444 |
| Swin-T (Swin Transformer Tiny) ❌ | Swin Transformer | 28M | 0 | 0.6198 | 0.6195 | 0.5366 | 0.6331 | 0.8112 | 0.6758 | 0.7204 |
| EfficientNet-B0 | CNN | 4M | N/A | N/A | 0.9984 | 0.9870 | N/A | 0.4400 | 0.5385 | N/A |

## Table 4.2 — OOD Per-Source Accuracy

| Model | camera_ai (Fake, N=88) | camera_real (Real, N=94) | Bias |
|-------|------------------------|--------------------------|------|
| ResNet-18 | 79.5% | 80.8% | False Negative dominant â€” misses some fakes |
| ViT-Small/16 | 76.1% | 73.4% | False Positive dominant â€” bias toward FAKE |
| Swin-T (Swin Transformer Tiny) | 86.4% | 50.0% | Extreme FAKE bias â€” classifies ~75% of ALL real images as fake |
| EfficientNet-B0 | 92.3% | 0.0% | N/A |

## Table 4.3 — ResNet-18 ID Per-Source Analysis (Best Model)

| Source | Type | Accuracy | N |
|--------|------|----------|---|
| camera_train_ai | Fake | 84.6% | 13 |
| camera_train_real | Fake | 78.6% | 14 |
| ciplab_training_fake | Fake | 96.4% | 192 |
| ciplab_training_real | Fake | 95.4% | 216 |
| dalle_fake | Fake | 99.0% | 200 |
| dd2025_fake | Fake | 96.2% | 500 |
| dd2025_real | Real | 97.6% | 500 |
| deepfake_collection_real | Fake | 98.0% | 589 |
| midjourney_fake | Fake | 100.0% | 93 |
| rvf10k_train_fake | Fake | 97.1% | 350 |
| rvf10k_train_real | Fake | 96.9% | 350 |
| rvf10k_valid_fake | Fake | 94.7% | 150 |
| rvf10k_valid_real | Real | 96.0% | 150 |
| sd_fake | Fake | 100.0% | 209 |

## Table 4.4 — Research Model Baselines (v2 test set — Fair Comparison)

> Re-benchmarked on v2 test sets for fair comparison with HolmHz models.

> Test set: data/manifests_v2/test_id.json (3526) + test_ood.json (182)

| Model | Architecture | ID AUC | ID Acc | OOD AUC | OOD Acc | Status |
|-------|-------------|--------|--------|---------|---------|--------|
| cnndetection | ResNet-50 (Wang et al. 2020) | 0.6619 | 0.5244 | 0.3253 | 0.5165 | Near random |
| universalfake | CLIP ViT-L/14 + Linear (Ojha et al. 2023) | 0.7218 | 0.7153 | 0.4858 | 0.5330 | Near random |
| deepfakebench | EfficientNet-B4 (Yan et al. 2023) | 0.4389 | 0.4504 | 0.5359 | 0.5385 | Near random |