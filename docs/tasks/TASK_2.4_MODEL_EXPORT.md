## 💡 Context

> **Task ID**: S2-004  
> **Phase**: Phase 1 - Data + Model Development  
> **Sprint**: Sprint 2 - Evaluation + XAI + Benchmark  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: 31/03/2026  
> **Assignee**: Hoàng  
> **Blocked by**: S1-006 (trained model)  
> **Blocks**: S3-002 (Web demo cần ONNX model)

> Export model sang ONNX format cho web inference tối ưu.
> Validate ONNX output khớp với PyTorch output.

---

## 🤖 AI Refined

> **User Story:**

> As a **ML Engineer**, I want to **export the trained model to ONNX format and validate correctness** so that **the web demo can use ONNX Runtime for optimized CPU inference (target ≤ 2s/image).**

**Acceptance Criteria:**

- [ ] PyTorch → ONNX export thành công (opset 17+)
- [ ] ONNX file size ≤ 50MB
- [ ] Validation: ONNX output matches PyTorch output (max diff < 1e-5) trên 10 test images
- [ ] ONNX Runtime inference benchmark: latency trên CPU
- [ ] Script `scripts/export_onnx.py` chạy 1 lệnh

---

## 🛠️ Implementation

### Subtasks

- [ ] 2.4.1 Implement `src/holmhz/export/onnx_export.py`
- [ ] 2.4.2 Implement `src/holmhz/export/validate.py` (compare PyTorch vs ONNX)
- [ ] 2.4.3 Script `scripts/export_onnx.py` CLI
- [ ] 2.4.4 Benchmark: measure inference latency (CPU)

### Branch & PR

- [ ] Branch: `feat/s2/onnx-export`
- [ ] PR Created
- [ ] Validation passed
- [ ] Latency benchmark documented

---

## 📝 Notes

> **Export snippet:**
>
> ```python
> import torch
> dummy_input = torch.randn(1, 3, 224, 224)
> torch.onnx.export(
>     model, dummy_input, "outputs/exports/holmhz_effb0.onnx",
>     opset_version=17,
>     input_names=["image"],
>     output_names=["probability"],
>     dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}}
> )
> ```

> **Latency target:**
>
> - CPU (Colab/local): ≤ 2s/image
> - EfficientNet-B0 ONNX thường ~100-300ms trên CPU → dư target
