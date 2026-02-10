## 💡 Context

> **Task ID**: S3-001  
> **Phase**: Phase 2 - Web Application & Report  
> **Sprint**: Sprint 3 - Web Demo Development  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: 14/04/2026  
> **Assignee**: Hoàng  
> **Blocked by**: S2-004 (ONNX model)  
> **Blocks**: S3-002 (Frontend cần API)

> Xây dựng FastAPI backend: predict, explain (Grad-CAM), health check.

---

## 🤖 AI Refined

> **User Story:**

> As a **Developer**, I want to **build a FastAPI backend with predict and explain endpoints** so that **the web demo can receive images, run inference, and return results with Grad-CAM heatmaps.**

**Acceptance Criteria:**

- [ ] `POST /api/predict`: nhận image → trả về `{probability, label, confidence}`
- [ ] `POST /api/explain`: nhận image → trả về heatmap overlay (base64 image)
- [ ] `GET /api/health`: trả về model loaded status
- [ ] Request validation: file type check (jpg/png), size limit (10MB)
- [ ] Error handling: invalid image, model not loaded
- [ ] Latency: predict ≤ 2s trên CPU
- [ ] Unit tests cho mỗi endpoint

---

## 🛠️ Implementation

### Subtasks

- [ ] 3.1.1 Setup FastAPI project trong `app/`
- [ ] 3.1.2 Implement `POST /api/predict` (load ONNX, preprocess, inference)
- [ ] 3.1.3 Implement `POST /api/explain` (Grad-CAM → heatmap → base64)
- [ ] 3.1.4 Implement `GET /api/health`
- [ ] 3.1.5 Request validation + error handling

### Branch & PR

- [ ] Branch: `feat/s3/backend-api`
- [ ] PR Created
- [ ] API tests passed (`pytest tests/test_api.py`)
- [ ] Latency benchmark: ≤ 2s confirmed

---

## 📝 Notes

> **Response schema:**
>
> ```json
> // POST /api/predict
> {
>   "probability": 0.87,
>   "label": "FAKE",
>   "confidence": "HIGH",
>   "processing_time_ms": 450
> }
>
> // POST /api/explain
> {
>   "probability": 0.87,
>   "label": "FAKE",
>   "heatmap_base64": "data:image/png;base64,iVBOR...",
>   "overlay_base64": "data:image/png;base64,iVBOR..."
> }
> ```
