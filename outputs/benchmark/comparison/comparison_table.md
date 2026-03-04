# Model Comparison — HolmHz Benchmark

> Generated: 2026-03-04 14:02

> Note: OOD test set is 100% fair (disjoint from all models' training data).
> ID test set contains 12.5% sources unique to HolmHz training.


## Overall Metrics (threshold=0.5)

| Model | ID AUC | ID Acc | OOD AUC | OOD Acc | OOD F1 |
| --- | --- | --- | --- | --- | --- |
| **cnndetection** | 0.5882 | 51.9% | 0.4264 | 44.0% | 0.0052 |
| **deepfakebench** | 0.5237 | 52.0% | 0.3913 | 42.8% | 0.4203 |
| **holmhz** | 0.9959 | 97.4% | 0.7823 | 71.3% | 0.7547 |
| **universalfake** | 0.6479 | 56.8% | 0.4674 | 44.1% | 0.0306 |

## Per-Source OOD Accuracy

| Model | flux | real_camera | real_pexels | tristanzhang_fake |
| --- | --- | --- | --- | --- |
| **cnndetection** | 0.0% | 100.0% | 99.0% | 0.3% |
| **deepfakebench** | 57.5% | 55.0% | 47.5% | 31.7% |
| **holmhz** | 77.5% | 36.0% | 74.5% | 79.3% |
| **universalfake** | 0.0% | 98.0% | 98.0% | 2.0% |
