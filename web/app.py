"""
HolmHz Web Demo — Synthetic Image Detector.

Upload an image to detect if it's AI-generated (Fake) or Real.
Powered by EfficientNet-B0 (4M params, OOD AUC 0.7838).

Usage:
    cd R:/_Projects/Eurus_Workspace/HolmHz
    source .venv/Scripts/activate
    python web/app.py

    → Open http://localhost:7860
"""

import sys
import os
import time
from pathlib import Path

# Add web/ + src/ to path
_web_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_web_dir))
sys.path.insert(0, str(_web_dir.parent / "src"))

# Must be before gradio import on Windows
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from model_service import OnnxPredictor, GradCAMService
from config import ONNX_MODEL_PATH, PYTORCH_CHECKPOINT, MODEL_NAME, DEVICE, THRESHOLD

# ──────────────────────────────────────────────────────────────
# Load models (once on startup)
# ──────────────────────────────────────────────────────────────
print("Loading ONNX predictor...", flush=True)
t0 = time.time()
predictor = OnnxPredictor(ONNX_MODEL_PATH, threshold=THRESHOLD)
print(f"  ONNX loaded in {time.time() - t0:.1f}s", flush=True)

print("Loading Grad-CAM service...", flush=True)
t1 = time.time()
gradcam_service = GradCAMService(MODEL_NAME, PYTORCH_CHECKPOINT, device=DEVICE)
print(f"  Grad-CAM loaded in {time.time() - t1:.1f}s", flush=True)
print("Models ready!\n", flush=True)


# ──────────────────────────────────────────────────────────────
# Prediction functions
# ──────────────────────────────────────────────────────────────
def predict_fn(image):
    """Tab 1: Quick predict — Real/Fake + confidence."""
    if image is None:
        return "No image uploaded", 0.0

    t = time.time()
    result = predictor.predict(image)
    elapsed = time.time() - t

    label_text = f"{'🚨 FAKE' if result['label'] == 'FAKE' else '✅ REAL'}"
    detail = (
        f"**{label_text}** — Confidence: {result['confidence']:.1%}\n\n"
        f"P(Fake) = {result['prob_fake']:.4f} | "
        f"Threshold = {THRESHOLD} | "
        f"Latency: {elapsed*1000:.0f}ms"
    )
    # For gr.Label: dict of {class: prob}
    label_dict = {"FAKE": result["prob_fake"], "REAL": 1 - result["prob_fake"]}
    return label_dict, detail


def explain_fn(image):
    """Tab 2: Predict + Grad-CAM heatmap."""
    if image is None:
        return "No image uploaded", 0.0, None

    t = time.time()
    result = predictor.predict(image)
    heatmap = gradcam_service.generate_heatmap(image)
    elapsed = time.time() - t

    label_text = f"{'🚨 FAKE' if result['label'] == 'FAKE' else '✅ REAL'}"
    detail = (
        f"**{label_text}** — Confidence: {result['confidence']:.1%}\n\n"
        f"P(Fake) = {result['prob_fake']:.4f} | "
        f"Threshold = {THRESHOLD} | "
        f"Latency: {elapsed*1000:.0f}ms\n\n"
        f"🔴 Red = Model focuses here | 🔵 Blue = Model ignores"
    )
    label_dict = {"FAKE": result["prob_fake"], "REAL": 1 - result["prob_fake"]}
    return label_dict, detail, heatmap


# ──────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────
css = """
.gradio-container { max-width: 900px; margin: auto; }
h1 { text-align: center; }
.footer { text-align: center; font-size: 0.85em; color: #888; margin-top: 1em; }
"""

with gr.Blocks(
    title="HolmHz — AI Image Detector",
    theme=gr.themes.Soft(),
    css=css,
) as demo:
    gr.Markdown(
        "# 🔍 HolmHz — Synthetic Image Detector\n"
        "Upload an image to detect if it's AI-generated (Fake) or Real.\n"
        "Powered by EfficientNet-B0 (4M params) trained on GAN + Diffusion data."
    )

    with gr.Tabs():
        # ── Tab 1: Quick Predict ──
        with gr.Tab("🎯 Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    predict_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        sources=["upload", "clipboard"],
                    )
                    predict_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

                with gr.Column(scale=1):
                    predict_label = gr.Label(label="Classification", num_top_classes=2)
                    predict_detail = gr.Markdown(label="Details")

            predict_btn.click(
                predict_fn,
                inputs=predict_input,
                outputs=[predict_label, predict_detail],
            )

            gr.Examples(
                examples=[
                    str(Path(_web_dir).parent / "imgs" / "Fake_AI_generated" / p.name)
                    for p in sorted((Path(_web_dir).parent / "imgs" / "Fake_AI_generated").iterdir())
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ][:3] + [
                    str(Path(_web_dir).parent / "imgs" / "Real" / p.name)
                    for p in sorted((Path(_web_dir).parent / "imgs" / "Real").iterdir())
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ][:3],
                inputs=predict_input,
                label="Example Images",
            )

        # ── Tab 2: Explain (Grad-CAM) ──
        with gr.Tab("🔬 Explain (Grad-CAM)"):
            with gr.Row():
                with gr.Column(scale=1):
                    explain_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        sources=["upload", "clipboard"],
                    )
                    explain_btn = gr.Button("🔬 Analyze + Explain", variant="primary", size="lg")

                with gr.Column(scale=1):
                    explain_label = gr.Label(label="Classification", num_top_classes=2)
                    explain_detail = gr.Markdown(label="Details")
                    explain_heatmap = gr.Image(label="Grad-CAM Heatmap", type="pil")

            explain_btn.click(
                explain_fn,
                inputs=explain_input,
                outputs=[explain_label, explain_detail, explain_heatmap],
            )

    gr.Markdown(
        '<div class="footer">'
        "<b>Model</b>: EfficientNet-B0 (4M params) | "
        "<b>Best OOD AUC</b>: 0.7838 | "
        "<b>Threshold</b>: 0.76 (Youden's J) | "
        "<b>ONNX</b>: 15.3 MB<br>"
        "HolmHz — AI-Generated Image Detection | Sprint 3 Demo"
        "</div>",
    )


# ──────────────────────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
