"""One-shot script: ONNX export (all 4 models) + Grad-CAM gallery generation.
Run: PYTHONUNBUFFERED=1 .venv/Scripts/python scripts/run_export_and_gradcam.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

print("=" * 60, flush=True)
print("HolmHz — ONNX Export + Grad-CAM Gallery", flush=True)
print("=" * 60, flush=True)

# ── Step 1: Import libs ──────────────────────────────────────────
t0 = time.time()
print("\n[1/5] Importing libraries...", flush=True)
import numpy as np
import torch
import onnxruntime as ort

from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors  # noqa: F401
from holmhz.exports.onnx_export import export_to_onnx
from holmhz.exports.validate import validate_onnx
from holmhz.xai.gradcam import GradCAMExplainer
from holmhz.xai.utils import load_image_for_gradcam

print(f"    Done in {time.time()-t0:.1f}s", flush=True)

# ── Step 2: ONNX Export ──────────────────────────────────────────
MODELS = [
    ("efficientnet_b0", "outputs/checkpoints/best_v4.pt"),
    ("resnet18", "outputs/checkpoints/best_resnet18.pt"),
    ("vit_small", "outputs/checkpoints/best_vit_small.pt"),
    ("swin_tiny", "outputs/checkpoints/best_swin_tiny.pt"),
]

export_dir = Path("outputs/exports")
export_dir.mkdir(parents=True, exist_ok=True)

print("\n[2/5] ONNX Export...", flush=True)
for model_name, ckpt_path in MODELS:
    if not Path(ckpt_path).exists():
        print(f"    SKIP {model_name}: {ckpt_path} not found", flush=True)
        continue

    t1 = time.time()
    print(f"    Exporting {model_name}...", flush=True)

    model = DETECTOR_REGISTRY.build(model_name, pretrained=False, freeze_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    onnx_path = export_dir / f"{model_name}.onnx"
    export_to_onnx(model, onnx_path, opset_version=17, input_shape=(1, 3, 224, 224), simplify=True)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"    [OK] {model_name} -> {onnx_path} ({size_mb:.1f} MB, {time.time()-t1:.1f}s)", flush=True)

    # Validate
    max_diff = validate_onnx(model, onnx_path, input_shape=(1, 3, 224, 224), tolerance=1e-5)
    print(f"      Validation: max_diff={max_diff:.2e}", flush=True)

    del model, ckpt, state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── Step 3: CPU Latency Benchmark (best model only) ──────────────
print("\n[3/5] CPU Latency Benchmark (EfficientNet-B0)...", flush=True)
onnx_path = export_dir / "efficientnet_b0.onnx"
if onnx_path.exists():
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: test_input})

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {input_name: test_input})
        times.append((time.perf_counter() - start) * 1000)

    print(f"    Mean:  {np.mean(times):.2f} ms", flush=True)
    print(f"    Std:   {np.std(times):.2f} ms", flush=True)
    print(f"    P50:   {np.percentile(times, 50):.2f} ms", flush=True)
    print(f"    P95:   {np.percentile(times, 95):.2f} ms", flush=True)

# ── Step 4: Grad-CAM Gallery ─────────────────────────────────────
print("\n[4/5] Grad-CAM Gallery...", flush=True)
gallery_dir = Path("outputs/xai_gallery")
gallery_dir.mkdir(parents=True, exist_ok=True)

# Image sources
image_sources = {
    "fake_ai": Path("imgs/Fake_AI_generated"),
    "real": Path("imgs/Real"),
}

# Add OOD sources if available
ood_base = Path("data/processed/ood_test")
for src in ["flux", "tristanzhang_fake", "real_pexels", "real_camera"]:
    src_dir = ood_base / src
    if src_dir.exists():
        image_sources[f"ood_{src}"] = src_dir

# Use best model only for gallery
best_model_name = "efficientnet_b0"
best_ckpt = "outputs/checkpoints/best_v4.pt"

model = DETECTOR_REGISTRY.build(best_model_name, pretrained=False, freeze_backbone=False)
ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)

explainer = GradCAMExplainer(model, device="cpu")
total_images = 0
exts = {".jpg", ".jpeg", ".png", ".webp"}

for source_name, source_dir in image_sources.items():
    if not source_dir.exists():
        print(f"    SKIP {source_name}: dir not found", flush=True)
        continue

    imgs = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in exts])
    # Take up to 10 per source for gallery
    imgs = imgs[:10]

    out_dir = gallery_dir / source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in imgs:
        try:
            tensor, rgb_image = load_image_for_gradcam(img_path)
            out_path = out_dir / f"gradcam_{img_path.stem}.png"
            explainer.save(tensor, rgb_image, out_path)
            total_images += 1
        except Exception as e:
            print(f"    ERROR {img_path.name}: {e}", flush=True)

    print(f"    [OK] {source_name}: {len(imgs)} images -> {out_dir}", flush=True)

print(f"    Total: {total_images} Grad-CAM images generated", flush=True)

# ── Step 5: Summary ──────────────────────────────────────────────
print("\n[5/5] Summary", flush=True)
print("=" * 60, flush=True)
print(f"ONNX exports:", flush=True)
for f in sorted(export_dir.glob("*.onnx")):
    print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB", flush=True)

print(f"\nGrad-CAM gallery: {total_images} images in {gallery_dir}", flush=True)
for d in sorted(gallery_dir.iterdir()):
    if d.is_dir():
        count = len(list(d.glob("*.png")))
        print(f"  {d.name}: {count} images", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
print("DONE.", flush=True)
