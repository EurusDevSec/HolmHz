## 💡 Context

> **Task ID**: S3-002  
> **Phase**: Phase 2 - Web Application & Report  
> **Sprint**: Sprint 3 - Web Demo Development  
> **Status**: ⬜ NOT STARTED  
> **Created**: 10/02/2026  
> **Target**: ~~21/04/2026~~ → **28/04/2026**  
> **Assignee**: Hoàng + Luân (test)  
> **Blocked by**: S3-001 (Backend API)  
> **Blocks**: Không  
> **Milestone**: ✅ M3 — Working web demo (latency ≤ 2s)

> Xây dựng Gradio frontend + integration + testing.

---

## 🤖 AI Refined

> **User Story:**

> As a **User**, I want to **upload an image through a web interface and see whether it's real or fake, with a confidence score and Grad-CAM heatmap** so that **I can understand AI-generated image detection visually.**

**Acceptance Criteria:**

- [ ] Gradio interface: image upload → result (Real/Fake, % confidence, heatmap overlay)
- [ ] UI components: upload area, result display, heatmap visualization, example images
- [ ] Connected to FastAPI backend (hoặc Gradio trực tiếp gọi model)
- [ ] End-to-end test: upload → predict → display ≤ 3s total
- [ ] Error handling: unsupported format, too large file, model error
- [ ] **Luân**: Test trên nhiều loại ảnh (selfie, landscape, AI generated), ghi feedback
- [ ] Deploy: chạy local hoặc Colab notebook

---

## 🛠️ Implementation

### Subtasks

- [ ] 3.2.1 Setup Gradio interface (`app/gradio_ui.py`)
- [ ] 3.2.2 Image upload + result display (Real/Fake + confidence bar)
- [ ] 3.2.3 Heatmap visualization tab
- [ ] 3.2.4 Example images gallery (từ test results)
- [ ] 3.2.5 UI styling và UX polish
- [ ] 3.2.6 End-to-end integration testing — Hoàng
- [ ] 3.2.7 User testing — **Luân**: thử nhiều ảnh, ghi feedback
- [ ] 3.2.8 Latency optimization (target ≤ 2s)
- [ ] 3.2.9 Deploy to local/Colab

### Branch & PR

- [ ] Branch: `feat/s3/frontend-ui`
- [ ] PR Created
- [ ] Demo video recorded
- [ ] Luân feedback addressed

---

## 📝 Notes

> **Gradio layout mẫu:**
>
> ```python
> import gradio as gr
>
> with gr.Blocks(title="HolmHz - Deepfake Detector") as demo:
>     gr.Markdown("# 🔍 HolmHz - Synthetic Image Detection")
>     with gr.Row():
>         with gr.Column():
>             input_image = gr.Image(type="pil", label="Upload Image")
>             submit_btn = gr.Button("Analyze", variant="primary")
>         with gr.Column():
>             label_output = gr.Label(label="Prediction")
>             confidence_output = gr.Number(label="Fake Probability")
>             heatmap_output = gr.Image(label="Grad-CAM Heatmap")
>     gr.Examples(examples=[...], inputs=input_image)
> ```

> **Fallback plan:**
> Nếu FastAPI + Gradio phức tạp → dùng Gradio standalone (gọi model trực tiếp, skip API layer).
> Proof-of-concept quan trọng hơn architecture hoàn hảo.
