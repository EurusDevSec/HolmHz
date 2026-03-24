# 📖 HƯỚNG DẪN CHI TIẾT TASK 2.4: MODEL EXPORT ONNX

> **Dành cho**: Lê Văn Hoàng — người chưa có nền tảng ML/DL, học qua thực hành  
> **Triết lý**: Mỗi bước không chỉ hướng dẫn **làm gì** mà giải thích **tại sao làm vậy**  
> **Thời gian**: ~30 phút (scripts đã có, chạy 1 lệnh)  
> **Tiền đề**: Task 2.2b Multi-Arch ✅ DONE, checkpoints đã train xong  
> **Tham chiếu**: [CONTEXT.md](../CONTEXT.md) Section 23 | [PROJECT_PLAN.md](../PROJECT_PLAN.md)  
>
> **Output**: 4 file `.onnx` trong `outputs/exports/`

---

## 📋 Mục lục

- [Bức tranh tổng thể](#bức-tranh-tổng-thể)
- [Tại sao cần ONNX?](#tại-sao-cần-onnx)
- [ONNX giải thích dễ hiểu](#onnx-giải-thích-dễ-hiểu)
- [Kiến trúc code đã implement](#kiến-trúc-code-đã-implement)
- [Cách chạy](#cách-chạy)
- [Kết quả export](#kết-quả-export)
- [Validation — PyTorch vs ONNX](#validation--pytorch-vs-onnx)
- [Sử dụng ONNX cho Web Demo](#sử-dụng-onnx-cho-web-demo)
- [Checklist hoàn thành](#checklist-hoàn-thành)
- [Troubleshooting](#troubleshooting)

---

## Bức tranh tổng thể

```
┌───────────────────────────────────────────────────────────┐
│                  DỰ ÁN HOLMHZ — SPRINT 2                  │
│                                                             │
│  Task 2.3  Grad-CAM XAI ✅                                 │
│                                                             │
│  ► Task 2.4  MODEL EXPORT ONNX  ✅ DONE                   │
│    │                                                        │
│    │  MỤC ĐÍCH:  Chuyển model PyTorch → ONNX              │
│    │  TẠI SAO:   ONNX nhanh hơn 2-5x khi inference        │
│    │             + Không cần cài PyTorch trên server       │
│    │             + Web demo dùng onnxruntime (nhẹ)         │
│    │                                                        │
│    │  OUTPUT: 4 file .onnx (15MB → 108MB)                  │
│    │                                                        │
│    └──► Sprint 3: Web Demo (dùng ONNX cho inference)      │
└───────────────────────────────────────────────────────────┘
```

---

## Tại sao cần ONNX?

```
┌──────────────── TẠI SAO EXPORT ONNX? ─────────────────────┐
│                                                            │
│  PyTorch Model (.pt):                                      │
│  • Cần cài PyTorch (~2GB) trên server                     │
│  • Import torch mất 5-10 giây                             │
│  • Inference chậm (Python overhead)                        │
│  • Cần GPU driver phù hợp                                 │
│                                                            │
│  ONNX Model (.onnx):                                      │
│  • Chỉ cần onnxruntime (~30MB) → NHẸ                     │
│  • Load nhanh (< 1 giây)                                  │
│  • Inference nhanh (C++ backend, optimized)               │
│  • Chạy được trên CPU/GPU, cross-platform                 │
│                                                            │
│  → Web Demo:                                              │
│  • Backend load efficientnet_b0.onnx (15MB)               │
│  • Inference ≤ 0.5s trên CPU                             │
│  • Triển khai dễ (Docker, Colab, Heroku...)               │
└────────────────────────────────────────────────────────────┘
```

---

## ONNX giải thích dễ hiểu

### ONNX là gì?

```
ONNX = Open Neural Network Exchange

Nó là một "ngôn ngữ chung" cho AI models:

 PyTorch Model (.pt)  ──export──► ONNX (.onnx) ──load──► onnxruntime
 TensorFlow Model     ──export──► ONNX (.onnx) ──load──► onnxruntime
 JAX Model            ──export──► ONNX (.onnx) ──load──► onnxruntime

Giống như PDF:
 • Word file  → Export PDF → Ai cũng đọc được
 • .pt file   → Export ONNX → Runtime nào cũng chạy được
```

### Quá trình export

```
 PyTorch Model
      ↓
 torch.onnx.export(model, dummy_input)
      ↓
 ONNX Graph (nodes + edges)
      ↓ (optional)
 onnxsim.simplify()  ← Tối ưu graph (gộp operations)
      ↓
 model.onnx file
      ↓
 validate_onnx()     ← So sánh PyTorch vs ONNX output
      ↓
 max_diff < 1e-4?  → ✅ PASS
```

---

## Kiến trúc code đã implement

### File structure

```
src/holmhz/exports/
├── __init__.py          # Exports export_to_onnx, validate_onnx
├── onnx_export.py       # export_to_onnx() function
└── validate.py          # validate_onnx() function

scripts/
├── export_onnx.py       # CLI: single model + CPU benchmark
└── export_all_onnx.py   # Batch: export tất cả 4 models
```

### `export_to_onnx()` function

```python
from holmhz.exports.onnx_export import export_to_onnx

# Export model sang ONNX
onnx_path = export_to_onnx(
    model,                           # PyTorch model đã load weights
    "outputs/exports/model.onnx",    # Output path
    opset_version=17,                # ONNX version (17 = stable + mới)
    input_shape=(1, 3, 224, 224),    # Batch=1, RGB, 224x224
    simplify=True,                   # Chạy onnxsim tối ưu
)
```

### `validate_onnx()` function

```python
from holmhz.exports.validate import validate_onnx

# So sánh PyTorch output vs ONNX output
max_diff = validate_onnx(
    model,                           # PyTorch model (gốc)
    "outputs/exports/model.onnx",    # ONNX file
    input_shape=(1, 3, 224, 224),
    tolerance=1e-4,                  # Sai số cho phép
)
# max_diff < 1e-4 → PASS ✅
```

---

## Cách chạy

### Bước 0: Activate venv

```bash
cd R:/_Projects/Eurus_Workspace/HolmHz
source .venv/Scripts/activate   # ← BẮT BUỘC!
```

### Bước 1: Export tất cả 4 models (khuyến nghị)

```bash
python -u scripts/export_all_onnx.py
```

Output:
```
============================================================
HolmHz ONNX Export — All 4 Models
============================================================

[1/3] Importing torch...
    torch imported in 7.7s
[2/3] Importing holmhz modules...
    holmhz imported in 6.0s

[3/3] Exporting models...
    --- efficientnet_b0 ---
    Model loaded (4.0M params)
    Exported: outputs/exports/efficientnet_b0.onnx (15.3 MB)
    Validated: max_diff=1.03e-05
    Done in 1.3s

    --- resnet18 ---
    Model loaded (11.2M params)
    Exported: outputs/exports/resnet18.onnx (42.6 MB)
    Validated: max_diff=2.50e-06
    Done in 0.9s

    --- vit_small ---
    Model loaded (21.7M params)
    Exported: outputs/exports/vit_small.onnx (82.8 MB)
    Validated: max_diff=5.01e-06
    Done in 2.1s

    --- swin_tiny ---
    Model loaded (27.5M params)
    Exported: outputs/exports/swin_tiny.onnx (107.6 MB)
    Validated: max_diff=3.10e-06
    Done in 3.7s

============================================================
EXPORT SUMMARY
============================================================
  efficientnet_b0: 15.3 MB, max_diff=1.03e-05, 1.3s
  resnet18: 42.6 MB, max_diff=2.50e-06, 0.9s
  vit_small: 82.8 MB, max_diff=5.01e-06, 2.1s
  swin_tiny: 107.6 MB, max_diff=3.10e-06, 3.7s

Total ONNX files: 4
Total time: 21.8s
```

### Bước 2: Single model + benchmark (optional)

```bash
python scripts/export_onnx.py configs/export.yaml --benchmark
```

---

## Kết quả export

### 4 ONNX files

| Model | Checkpoint | ONNX File | Size | max_diff | Status |
| --- | --- | --- | --- | --- | --- |
| **EfficientNet-B0** | `best_v4.pt` | `efficientnet_b0.onnx` | **15.3 MB** | 1.03e-05 | ✅ |
| **ResNet-18** | `best_resnet18.pt` | `resnet18.onnx` | 42.6 MB | 2.50e-06 | ✅ |
| **ViT-Small/16** | `best_vit_small.pt` | `vit_small.onnx` | 82.8 MB | 5.01e-06 | ✅ |
| **Swin-T** | `best_swin_tiny.pt` | `swin_tiny.onnx` | 107.6 MB | 3.10e-06 | ✅ |

### File structure

```
outputs/exports/
├── efficientnet_b0.onnx   ← 15.3 MB  ← CHO WEB DEMO
├── resnet18.onnx           ← 42.6 MB
├── vit_small.onnx          ← 82.8 MB
├── swin_tiny.onnx          ← 107.6 MB
└── export_log.txt          ← Log file
```

---

## Validation — PyTorch vs ONNX

### Tại sao cần validate?

```
Export ONNX = Dịch model sang ngôn ngữ khác.
Phải kiểm tra: bản dịch có chính xác không?

PyTorch model: predict Fake 87.32%
ONNX model:    predict Fake 87.32%  ← PHẢI GIỐNG!

max_diff = |87.32% - 87.32%| ≈ 0.00001 → OK!
```

### Ngưỡng chấp nhận

```
max_diff < 1e-4 (0.0001) → ✅ PASS (floating point precision)
max_diff > 1e-3 (0.001)  → ⚠️ WARNING (có thể do ops không supported)
max_diff > 1e-2 (0.01)   → ❌ FAIL (export bị lỗi)
```

### Kết quả validation (24/03/2026)

```
All 4 models: max_diff < 1e-4 → ✅ PASS
Best:  ResNet-18      (2.50e-06 = 0.0000025 → gần như identical)
Worst: EfficientNet-B0 (1.03e-05 = 0.0000103 → vẫn rất tốt)
```

---

## Sử dụng ONNX cho Web Demo

### Sprint 3 sẽ dùng ONNX như thế nào?

```python
# Backend API (Task 3.1.2 Model Service)
import onnxruntime as ort

# 1. Load model (1 lần khi server start)
session = ort.InferenceSession("outputs/exports/efficientnet_b0.onnx")

# 2. Inference (mỗi request)
import numpy as np
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {"input": input_data})
prob_fake = 1 / (1 + np.exp(-output[0][0][0]))  # sigmoid
```

### So sánh hiệu năng

| Metric | PyTorch (.pt) | ONNX (.onnx) |
| --- | --- | --- |
| **Model load** | ~8s (import torch + load) | ~0.5s (onnxruntime) |
| **Inference** | ~50ms (GPU) / ~200ms (CPU) | ~30ms (GPU) / ~100ms (CPU) |
| **Dependencies** | torch (2GB) | onnxruntime (30MB) |
| **Docker image** | ~5GB | **~500MB** |

---

## Checklist hoàn thành

- [x] `src/holmhz/exports/onnx_export.py` — export_to_onnx() ✅
- [x] `src/holmhz/exports/validate.py` — validate_onnx() ✅
- [x] `scripts/export_onnx.py` — Single model CLI ✅
- [x] `scripts/export_all_onnx.py` — Batch export ✅
- [x] `outputs/exports/efficientnet_b0.onnx` (15.3 MB) ✅
- [x] `outputs/exports/resnet18.onnx` (42.6 MB) ✅
- [x] `outputs/exports/vit_small.onnx` (82.8 MB) ✅
- [x] `outputs/exports/swin_tiny.onnx` (107.6 MB) ✅
- [x] All 4 models: max_diff < 1e-4 ✅
- [x] CONTEXT.md Section 23 updated ✅
- [x] PROJECT_PLAN.md Task 2.4 → ✅ ✅

---

## Troubleshooting

### `ModuleNotFoundError: onnxsim`

```bash
pip install onnxsim
```

Hoặc bỏ `simplify=True` — model vẫn hoạt động, chỉ lớn hơn một chút.

### `RuntimeError: Exporting the operator ... not supported`

Một số PyTorch operations không có ONNX equivalent. Workaround:
- Giảm `opset_version` (thử 14 hoặc 11)
- Hoặc viết custom op wrapper

### `AssertionError: max_diff > tolerance`

ONNX output khác PyTorch quá nhiều. Nguyên nhân có thể:
- `onnxsim` simplify quá mạnh → thử `simplify=False`
- Model dùng operations không deterministic (dropout, batch norm running stats)
- Tăng `tolerance` (1e-4 → 1e-3) nếu sai số nhỏ

### `import torch` bị hang/rất chậm

**⚠️ BẮT BUỘC activate venv trước:**
```bash
source .venv/Scripts/activate
```
Nếu không activate venv, `import torch` sẽ hang vô hạn trên Windows.
