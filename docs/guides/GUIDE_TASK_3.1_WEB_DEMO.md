# 📖 HƯỚNG DẪN CHI TIẾT TASK 3.1: WEB DEMO (BACKEND + FRONTEND)

> **Dành cho**: Lê Văn Hoàng — người chưa có nền tảng ML/DL, học qua thực hành  
> **Triết lý**: Mỗi bước không chỉ hướng dẫn **làm gì** mà giải thích **tại sao làm vậy**  
> **Thời gian**: ~3-5 ngày  
> **Tiền đề**: Sprint 2 ✅ DONE (ONNX exported, Grad-CAM implemented)  
> **Tham chiếu**: [CONTEXT.md](../CONTEXT.md) | [PROJECT_PLAN.md](../PROJECT_PLAN.md) Sprint 3  
>
> **Output**: Web app hoạt động tại `http://localhost:7860` — upload ảnh → Real/Fake + confidence + heatmap

---

## 📋 Mục lục

- [Bức tranh tổng thể](#bức-tranh-tổng-thể)
- [Tại sao Gradio?](#tại-sao-gradio)
- [Kiến trúc Web Demo](#kiến-trúc-web-demo)
- [Task 3.1.1 — Backend API](#task-311--backend-api)
- [Task 3.1.2 — Model Service](#task-312--model-service)
- [Task 3.1.3 — Frontend UI (Gradio)](#task-313--frontend-ui-gradio)
- [Task 3.1.5 — Integration & Deploy](#task-315--integration--deploy)
- [Cách test](#cách-test)
- [Checklist hoàn thành](#checklist-hoàn-thành)
- [Troubleshooting](#troubleshooting)

---

## Bức tranh tổng thể

```
┌───────────────────────────────────────────────────────────────┐
│                   DỰ ÁN HOLMHZ — SPRINT 3                    │
│                                                                │
│  Sprint 1-2 ✅ HOÀN TẤT                                      │
│  • EfficientNet-B0: OOD AUC 0.7838 (Best of 7 models)       │
│  • ONNX exported: efficientnet_b0.onnx (15.3 MB)            │
│  • Grad-CAM: 61 heatmap images                              │
│                                                                │
│  ► Sprint 3: WEB DEMO                                        │
│    │                                                          │
│    ├── 3.1.1 Backend API (FastAPI)                            │
│    │    • POST /api/predict → Real/Fake + confidence         │
│    │    • POST /api/explain → Grad-CAM heatmap               │
│    │    • GET  /api/health  → Server status                  │
│    │                                                          │
│    ├── 3.1.2 Model Service                                   │
│    │    • Load ONNX model on startup                         │
│    │    • Preprocessing pipeline                             │
│    │    • Grad-CAM service (PyTorch model)                   │
│    │                                                          │
│    ├── 3.1.3 Frontend UI (Gradio)                            │
│    │    • Image upload                                       │
│    │    • Result: Real/Fake + confidence bar                 │
│    │    • Heatmap visualization                              │
│    │                                                          │
│    ├── 3.1.4 UI Testing (Luân)                               │
│    │                                                          │
│    └── 3.1.5 Integration & Deploy                            │
│         • Latency target: ≤ 2s/ảnh                          │
│         • Deploy: local + Colab + (maybe) HuggingFace       │
└───────────────────────────────────────────────────────────────┘
```

---

## Tại sao Gradio?

```
┌──────────────── SO SÁNH FRAMEWORK ────────────────────────┐
│                                                            │
│  Gradio:                                                   │
│  ✅ 1 file Python = cả Frontend + Backend                 │
│  ✅ Có sẵn image upload, gallery, labels                  │
│  ✅ Tự tạo API endpoint (cho Luân test riêng)             │
│  ✅ Deploy HuggingFace Spaces (FREE, 1 click)             │
│  ✅ Không cần biết HTML/CSS/JS                            │
│  ⚡ 30 phút là có demo chạy được                          │
│                                                            │
│  Streamlit:                                                │
│  ✅ Cũng dễ dùng                                          │
│  ❌ Không có built-in image upload tốt                    │
│  ❌ Không tự tạo REST API                                 │
│                                                            │
│  React + FastAPI:                                          │
│  ✅ Professional nhất                                     │
│  ❌ Cần 2 projects riêng (frontend + backend)             │
│  ❌ Mất 1-2 tuần                                         │
│  ❌ Overkill cho thesis demo                              │
│                                                            │
│  → CHỌN GRADIO: nhanh, đủ tính năng, dễ deploy.          │
└────────────────────────────────────────────────────────────┘
```

---

## Kiến trúc Web Demo

```
Người dùng (browser)
    │
    │  Upload ảnh (drag & drop / click)
    ▼
┌──────────────────┐
│  Gradio Frontend │  ← Tab 1: Predict, Tab 2: Explain
│  (Python)        │
└────────┬─────────┘
         │  Call Python function trực tiếp
         ▼
┌──────────────────────────────────────────────┐
│  Model Service (Python module)                │
│                                                │
│  ┌─────────────────────────────────────────┐   │
│  │ OnnxPredictor                            │   │
│  │  • Load efficientnet_b0.onnx (1 lần)    │   │
│  │  • preprocess(): resize + normalize      │   │
│  │  • predict(): prob_fake, label           │   │
│  └─────────────────────────────────────────┘   │
│                                                │
│  ┌─────────────────────────────────────────┐   │
│  │ GradCAMService                           │   │
│  │  • Load PyTorch model (1 lần)            │   │
│  │  • generate_heatmap(): overlay image     │   │
│  └─────────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
         │
         ▼
    Hiển thị kết quả
    • "FAKE" / "REAL" + confidence %
    • Grad-CAM heatmap overlay
```

> **Lưu ý**: Predict dùng ONNX (nhanh), Explain dùng PyTorch (cần gradients cho Grad-CAM).

---

## Task 3.1.1 — Backend API

### Cấu trúc files dự kiến

```
web/
├── app.py               # Gradio web application (main entry)
├── model_service.py     # OnnxPredictor + GradCAMService
├── config.py            # Paths, thresholds, settings
└── requirements.txt     # Web-specific dependencies
```

### Dependencies

```
# web/requirements.txt
gradio>=4.0
onnxruntime>=1.16
Pillow
numpy
# Grad-CAM needs PyTorch (already installed)
```

### Endpoints (auto-generated by Gradio)

| Endpoint | Method | Input | Output |
| --- | --- | --- | --- |
| `/api/predict` | POST | Image file | `{label, confidence, prob_fake}` |
| `/api/explain` | POST | Image file | `{label, confidence, heatmap_image}` |
| `/` | GET | - | Gradio UI (HTML) |

> Gradio tự tạo REST API từ Python functions — không cần viết FastAPI riêng.

---

## Task 3.1.2 — Model Service

### OnnxPredictor — Inference nhanh

```python
# web/model_service.py

import numpy as np
import onnxruntime as ort
from PIL import Image

class OnnxPredictor:
    """Predict Real/Fake using ONNX model."""

    def __init__(self, model_path: str, threshold: float = 0.76):
        self.session = ort.InferenceSession(model_path)
        self.threshold = threshold  # Youden's J optimal from v4
        self.input_name = self.session.get_inputs()[0].name

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """PIL Image → [1, 3, 224, 224] normalized array."""
        img = image.resize((224, 224)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW
        arr = (arr - self.mean) / self.std
        return arr.astype(np.float32)

    def predict(self, image: Image.Image) -> dict:
        """Predict Real/Fake + confidence."""
        input_arr = self.preprocess(image)
        logit = self.session.run(None, {self.input_name: input_arr})[0][0][0]
        prob_fake = float(1 / (1 + np.exp(-logit)))  # sigmoid

        label = "FAKE" if prob_fake >= self.threshold else "REAL"
        confidence = prob_fake if label == "FAKE" else 1 - prob_fake

        return {
            "label": label,
            "confidence": float(confidence),
            "prob_fake": float(prob_fake),
        }
```

### GradCAMService — Heatmap generation

```python
class GradCAMService:
    """Generate Grad-CAM heatmap (requires PyTorch)."""

    def __init__(self, model_name: str, checkpoint_path: str, device: str = "cpu"):
        import torch
        from holmhz.utils.registry import DETECTOR_REGISTRY
        import holmhz.detectors  # trigger registration

        self.model = DETECTOR_REGISTRY.build(model_name, pretrained=False, freeze_backbone=False)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.model.eval()

        from holmhz.xai.gradcam import GradCAMExplainer
        self.explainer = GradCAMExplainer(self.model, device=device)

    def generate_heatmap(self, image: Image.Image) -> Image.Image:
        """Generate Grad-CAM overlay image."""
        from holmhz.xai.utils import load_image_for_gradcam
        import tempfile, os

        # Save PIL → temp file → load with gradcam utils
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            tensor, rgb_image = load_image_for_gradcam(f.name)
            os.unlink(f.name)

        overlay = self.explainer.overlay(tensor, rgb_image)
        return Image.fromarray(overlay)
```

---

## Task 3.1.3 — Frontend UI (Gradio)

### Main application

```python
# web/app.py

import gradio as gr
from model_service import OnnxPredictor, GradCAMService

# ── Load models (1 lần khi start) ──
predictor = OnnxPredictor("outputs/exports/efficientnet_b0.onnx")
gradcam = GradCAMService("efficientnet_b0", "outputs/checkpoints/best_v4.pt")

def predict_fn(image):
    """Tab 1: Predict Real/Fake."""
    result = predictor.predict(image)
    label = f"{result['label']} ({result['confidence']:.1%})"
    return label, result["prob_fake"]

def explain_fn(image):
    """Tab 2: Predict + Grad-CAM heatmap."""
    result = predictor.predict(image)
    heatmap = gradcam.generate_heatmap(image)
    label = f"{result['label']} ({result['confidence']:.1%})"
    return label, result["prob_fake"], heatmap

# ── Gradio Interface ──
with gr.Blocks(title="HolmHz — AI Image Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 HolmHz — Synthetic Image Detector")
    gr.Markdown("Upload an image to detect if it's AI-generated (Fake) or Real.")

    with gr.Tabs():
        with gr.Tab("🎯 Predict"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Upload Image")
                    predict_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    result_label = gr.Label(label="Result")
                    prob_slider = gr.Slider(0, 1, label="P(Fake)", interactive=False)

            predict_btn.click(predict_fn, inputs=input_img, outputs=[result_label, prob_slider])

        with gr.Tab("🔬 Explain (Grad-CAM)"):
            with gr.Row():
                with gr.Column():
                    explain_img = gr.Image(type="pil", label="Upload Image")
                    explain_btn = gr.Button("Analyze + Explain", variant="primary")
                with gr.Column():
                    explain_label = gr.Label(label="Result")
                    explain_prob = gr.Slider(0, 1, label="P(Fake)", interactive=False)
                    heatmap_out = gr.Image(label="Grad-CAM Heatmap")

            explain_btn.click(explain_fn, inputs=explain_img,
                            outputs=[explain_label, explain_prob, heatmap_out])

    gr.Markdown("---")
    gr.Markdown("**Model**: EfficientNet-B0 (4M params) | **OOD AUC**: 0.7838 | **Threshold**: 0.76")

demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## Task 3.1.5 — Integration & Deploy

### Chạy local

```bash
cd R:/_Projects/Eurus_Workspace/HolmHz
source .venv/Scripts/activate
python web/app.py
# → Open http://localhost:7860
```

### Latency target

```
Upload → Preprocess → ONNX Inference → Display:  ≤ 1s (target)
Upload → Preprocess → Grad-CAM → Display:        ≤ 3s (acceptable)
```

### Deploy options

| Option | Pros | Cons |
| --- | --- | --- |
| **Local** | Nhanh nhất, debug dễ | Chỉ dùng 1 máy |
| **Google Colab** | Free GPU, share link | Không persistent |
| **HuggingFace Spaces** | Free, persistent URL | Model upload 15MB |
| **Docker** | Professional, reproducible | Cần setup |

---

## Cách test

### Test nhanh

```bash
# 1. Start server
python web/app.py

# 2. Open browser → http://localhost:7860
# 3. Upload ảnh từ imgs/Fake_AI_generated/ → expect "FAKE"
# 4. Upload ảnh từ imgs/Real/ → expect "REAL"
# 5. Click tab "Explain" → upload ảnh → xem heatmap
```

### Test API (cho Luân)

```bash
# Predict
curl -X POST http://localhost:7860/api/predict -F "image=@imgs/test.png"

# Explain
curl -X POST http://localhost:7860/api/explain -F "image=@imgs/test.png"
```

---

## Checklist hoàn thành

### Task 3.1.1 — Backend API
- [ ] `web/app.py` — Gradio application
- [ ] `web/config.py` — Configuration
- [ ] POST /api/predict working
- [ ] POST /api/explain working
- [ ] GET / returns Gradio UI

### Task 3.1.2 — Model Service
- [ ] `web/model_service.py` — OnnxPredictor + GradCAMService
- [ ] ONNX model loads on startup
- [ ] Preprocessing pipeline correct (ImageNet norm)
- [ ] Grad-CAM generates heatmap
- [ ] Error handling for invalid images

### Task 3.1.3 — Frontend UI
- [ ] Image upload (drag & drop + click)
- [ ] Result display: REAL/FAKE + confidence %
- [ ] Grad-CAM heatmap tab
- [ ] Mobile-responsive layout
- [ ] Loading indicator

### Task 3.1.5 — Integration
- [ ] End-to-end test: upload → result ≤ 2s
- [ ] Error cases: invalid file type, empty upload
- [ ] Deploy to local (verified)
- [ ] Deploy to Colab/HuggingFace (optional)

---

## Troubleshooting

### `ModuleNotFoundError: gradio`

```bash
pip install gradio
```

### Gradio UI không hiển thị

Kiểm tra firewall/port. Thử:
```python
demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
```
`share=True` tạo public URL qua Gradio tunnel.

### Heatmap chậm (> 5s)

Grad-CAM cần backward pass qua PyTorch → chậm hơn predict.
Giải pháp: chỉ chạy Grad-CAM khi user click "Explain", không tự động.

### ONNX inference cho kết quả khác PyTorch

Kiểm tra:
1. Preprocessing giống nhau (ImageNet norm, resize 224)
2. Sigmoid applied correctly
3. Threshold = 0.76 (Youden's J optimal)
