# KPI Assessment — HolmHz Project

> Assessed: 2026-04-09
> Reference: plan.md Section 9 (Mục tiêu đề tài)

## Summary

| KPI | Target | Actual | Status |
|-----|--------|--------|--------|
| Dataset size | ≥ 20,000 | 28,220 | ✅ |
| ID AUC ≥ 0.92 | ≥ 0.92 | 0.9984 (EfficientNet-B0) | ✅ |
| ID Accuracy ≥ 90% | ≥ 0.9 | 0.9870 (EfficientNet-B0) | ✅ |
| ID F1 ≥ 0.90 | ≥ 0.9 | 0.9702 (ResNet-18) | ✅ |
| OOD AUC ≥ 0.85 | ≥ 0.85 | 0.8646 (ResNet-18) | ✅ |

## Per-Model KPI Breakdown

| Model | ID AUC ≥0.92 | ID Acc ≥90% | ID F1 ≥0.90 | OOD AUC ≥0.85 | Overall |
|-------|--------------|-------------|-------------|---------------|---------|
| EfficientNet-B0 | ✅ | ✅ | ❌ | ❌ | 2/4 |
| ResNet-18 | ✅ | ✅ | ✅ | ✅ | 4/4 |
| ViT-Small/16 | ✅ | ✅ | ✅ | ❌ | 3/4 |
| Swin-T (Swin Transformer Tiny) | ❌ | ❌ | ❌ | ❌ | 0/4 ⚠️ FAILED |

## Verdict

**ResNet-18 đạt tất cả KPIs** — ID AUC 0.9953, OOD AUC 0.8646 ✅

### Model Ranking (by OOD generalization)

1. **ResNet-18** — Best overall: ID AUC 0.9953, OOD AUC 0.8646 (meets all KPIs)
2. **ViT-Small/16** — Good ID (0.9741), OOD close to target (0.8331 < 0.85)
3. **EfficientNet-B0 v7** — Best ID (0.9984) but worst OOD (0.44 = anti-correlated)
4. **Swin-T** — Training FAILED (best epoch = 0, ID AUC 0.62)

### Key Findings

1. **ResNet-18 is the best model for this task** — simpler CNN architecture generalizes better than larger transformers on this dataset size (~28K images).
2. **Model complexity ≠ better generalization** — Swin-T (28M) failed entirely, while ResNet-18 (11M) excelled. ViT-Small/16 (22M) was middle-ground.
3. **EfficientNet-B0 overfits to training distribution** — Excellent ID but inverted OOD predictions suggest it learned dataset-specific artifacts, not universal fake markers.
4. **OOD test set is very small (182 samples)** — Results should be validated on a larger external test set for statistical significance.