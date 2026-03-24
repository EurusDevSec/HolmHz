"""Export all 4 HolmHz models to ONNX format.
Simpler script — exports one model at a time, minimal imports.

Usage: .venv/Scripts/python scripts/export_all_onnx.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

print("=" * 60)
print("HolmHz ONNX Export — All 4 Models")
print("=" * 60)

# Step 1: Import torch
t0 = time.time()
print("\n[1/3] Importing torch...", flush=True)
import torch
print(f"    torch imported in {time.time()-t0:.1f}s", flush=True)

# Step 2: Import holmhz
t1 = time.time()
print("[2/3] Importing holmhz modules...", flush=True)
from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors  # noqa: F401
from holmhz.exports.onnx_export import export_to_onnx
from holmhz.exports.validate import validate_onnx
print(f"    holmhz imported in {time.time()-t1:.1f}s", flush=True)

# Step 3: Export each model
MODELS = [
    ("efficientnet_b0", "outputs/checkpoints/best_v4.pt"),
    ("resnet18", "outputs/checkpoints/best_resnet18.pt"),
    ("vit_small", "outputs/checkpoints/best_vit_small.pt"),
    ("swin_tiny", "outputs/checkpoints/best_swin_tiny.pt"),
]

export_dir = Path("outputs/exports")
export_dir.mkdir(parents=True, exist_ok=True)

print("\n[3/3] Exporting models...", flush=True)
results = []

for model_name, ckpt_path in MODELS:
    if not Path(ckpt_path).exists():
        print(f"    SKIP {model_name}: checkpoint not found", flush=True)
        continue

    t2 = time.time()
    print(f"\n    --- {model_name} ---", flush=True)

    # Build model
    model = DETECTOR_REGISTRY.build(model_name, pretrained=False, freeze_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"    Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)", flush=True)

    # Export ONNX
    onnx_path = export_dir / f"{model_name}.onnx"
    export_to_onnx(model, onnx_path, opset_version=17, input_shape=(1, 3, 224, 224), simplify=True)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"    Exported: {onnx_path} ({size_mb:.1f} MB)", flush=True)

    # Validate
    max_diff = validate_onnx(model, onnx_path, input_shape=(1, 3, 224, 224), tolerance=1e-4)
    print(f"    Validated: max_diff={max_diff:.2e}", flush=True)

    elapsed = time.time() - t2
    print(f"    Done in {elapsed:.1f}s", flush=True)
    results.append((model_name, size_mb, max_diff, elapsed))

    # Free memory
    del model, ckpt, state_dict

# Summary
print("\n" + "=" * 60)
print("EXPORT SUMMARY")
print("=" * 60)
for name, size, diff, elapsed in results:
    print(f"  {name}: {size:.1f} MB, max_diff={diff:.2e}, {elapsed:.1f}s")
print(f"\nTotal ONNX files: {len(results)}")
print(f"Total time: {time.time()-t0:.1f}s")
