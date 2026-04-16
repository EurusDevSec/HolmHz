"""
HolmHz Web Demo — Synthetic Image Detector.

Upload an image to detect if it's AI-generated (Fake) or Real.
Primary model: ResNet-18 v2 (ID AUC 0.9953, OOD AUC 0.8646).
Optional CLIP ensemble for additional OOD robustness.

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
from model_service import OnnxPredictor, CLIPPredictor, EnsemblePredictor, GradCAMService
from config import (
    ONNX_MODEL_PATH, PYTORCH_CHECKPOINT, MODEL_NAME, DEVICE, THRESHOLD,
    CLIP_CHECKPOINT, EFFNET_WEIGHT, CLIP_WEIGHT,
)

# ──────────────────────────────────────────────────────────────
# Load models (once on startup)
# ──────────────────────────────────────────────────────────────
print(f"Loading ONNX predictor ({MODEL_NAME})...", flush=True)
t0 = time.time()
effnet_predictor = OnnxPredictor(ONNX_MODEL_PATH, threshold=THRESHOLD)
print(f"  {MODEL_NAME} loaded in {time.time() - t0:.1f}s", flush=True)

# Load CLIP (optional — graceful fallback)
clip_predictor = None
if Path(CLIP_CHECKPOINT).exists():
    try:
        print("Loading CLIP ViT-L/14 predictor...", flush=True)
        t1 = time.time()
        clip_predictor = CLIPPredictor(CLIP_CHECKPOINT, threshold=THRESHOLD, device=DEVICE)
        print(f"  CLIP loaded in {time.time() - t1:.1f}s", flush=True)
    except Exception as e:
        print(f"  ⚠️ CLIP load failed: {e}", flush=True)
        print(f"  Falling back to {MODEL_NAME} only.", flush=True)
else:
    print(f"  ⚠️ CLIP checkpoint not found: {CLIP_CHECKPOINT}", flush=True)
    print("  Running EfficientNet only mode.", flush=True)

# Load EXIF analyzer
from holmhz.analysis.exif_analyzer import EXIFAnalyzer
exif_analyzer = EXIFAnalyzer(
    camera_multiplier=0.5,     # Camera EXIF → p_fake *= 0.5
    gps_multiplier=0.85,       # + GPS → p_fake *= 0.85
    ai_software_multiplier=1.2,  # AI software → p_fake *= 1.2
)
print("  ✅ EXIF analyzer loaded", flush=True)

# Ensemble predictor
predictor = EnsemblePredictor(
    effnet_predictor=effnet_predictor,
    clip_predictor=clip_predictor,
    exif_analyzer=exif_analyzer,
    threshold=THRESHOLD,
    effnet_weight=EFFNET_WEIGHT,
    clip_weight=CLIP_WEIGHT,
)
has_clip = clip_predictor is not None
ensemble_mode = "Ensemble + EXIF" if has_clip else "EfficientNet + EXIF"
print(f"\n🚀 Mode: {ensemble_mode}", flush=True)

print("Loading Grad-CAM service...", flush=True)
t2 = time.time()
gradcam_service = GradCAMService(MODEL_NAME, PYTORCH_CHECKPOINT, device=DEVICE)
print(f"  Grad-CAM loaded in {time.time() - t2:.1f}s", flush=True)
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

    # Build detail string
    detail_parts = [
        f"**{label_text}** — Confidence: {result['confidence']:.1%}\n\n",
        f"P(Fake) = {result['prob_fake']:.4f} | ",
        f"Threshold = {THRESHOLD} | ",
        f"Latency: {elapsed*1000:.0f}ms\n\n",
    ]

    # Show model probabilities
    if result.get("clip_prob") is not None:
        detail_parts.append(
            f"📊 **Ensemble** — "
            f"EfficientNet: {result['effnet_prob']:.4f} | "
            f"CLIP: {result['clip_prob']:.4f} | "
            f"Weights: {EFFNET_WEIGHT:.0%}/{CLIP_WEIGHT:.0%}\n\n"
        )
    else:
        detail_parts.append("📊 EfficientNet only\n\n")

    # EXIF info
    if result.get("exif_multiplier", 1.0) != 1.0:
        detail_parts.append(
            f"🔑 **EXIF**: {result['exif_summary']} "
            f"(raw: {result.get('prob_fake_raw', 0):.4f} → adjusted: {result['prob_fake']:.4f})"
        )
    elif result.get("exif_summary"):
        detail_parts.append(f"🔑 EXIF: {result['exif_summary']}")

    detail = "".join(detail_parts)
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

    detail_parts = [
        f"**{label_text}** — Confidence: {result['confidence']:.1%}\n\n",
        f"P(Fake) = {result['prob_fake']:.4f} | ",
        f"Threshold = {THRESHOLD} | ",
        f"Latency: {elapsed*1000:.0f}ms\n\n",
    ]

    if result.get("clip_prob") is not None:
        detail_parts.append(
            f"📊 **Ensemble** — "
            f"EfficientNet: {result['effnet_prob']:.4f} | "
            f"CLIP: {result['clip_prob']:.4f}\n\n"
        )

    # EXIF info
    if result.get("exif_multiplier", 1.0) != 1.0:
        detail_parts.append(
            f"🔑 **EXIF**: {result['exif_summary']} "
            f"(raw: {result.get('prob_fake_raw', 0):.4f} → {result['prob_fake']:.4f})\n\n"
        )
    elif result.get("exif_summary"):
        detail_parts.append(f"🔑 EXIF: {result['exif_summary']}\n\n")

    detail_parts.append("🔴 Red = Model focuses here | 🔵 Blue = Model ignores")

    detail = "".join(detail_parts)
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
        f"**Mode: {ensemble_mode}** — EfficientNet-B0 (ID) + CLIP ViT-L/14 (OOD)"
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
        f"<b>Mode</b>: {ensemble_mode} | "
        "<b>EfficientNet</b>: v8 (ID AUC 0.9984) | "
        "<b>CLIP</b>: v9 (OOD AUC 0.9419) | "
        f"<b>Weights</b>: {EFFNET_WEIGHT:.0%}/{CLIP_WEIGHT:.0%} | "
        f"<b>Threshold</b>: {THRESHOLD}<br>"
        "HolmHz — AI-Generated Image Detection | Ensemble v10"
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
