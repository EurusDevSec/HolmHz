"""
Generate Grad-CAM gallery — 50 samples from all OOD sources + train samples.

Sources:
  - ood_test/flux (all 80 → pick 10)
  - ood_test/tristanzhang_fake (300 → pick 15)
  - ood_test/real_pexels (200 → pick 10)
  - ood_test/real_camera (100 → pick 10)
  - data/processed/train/fake_diffusion/sd15 (→ pick 5)

Total target: ~50 images

Usage:
    python scripts/generate_xai_gallery.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors  # noqa: F401  -- trigger registration
from holmhz.xai.gradcam import GradCAMExplainer
from holmhz.xai.utils import load_image_for_gradcam

random.seed(42)

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
CHECKPOINT = "outputs/checkpoints/best_v4.pt"
MODEL_NAME = "efficientnet_b0"
OUTPUT_DIR = Path("outputs/xai_gallery")
DEVICE = "cpu"

SOURCES = [
    ("data/processed/ood_test/flux", "flux", 10),
    ("data/processed/ood_test/tristanzhang_fake", "tristanzhang_fake", 15),
    ("data/processed/ood_test/real_pexels", "real_pexels", 10),
    ("data/processed/ood_test/real_camera", "real_camera", 10),
    ("data/processed/train/fake_diffusion/sd15", "sd15_fake", 5),
]

EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# ──────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} from {CHECKPOINT}", flush=True)
model = DETECTOR_REGISTRY.build(MODEL_NAME, pretrained=False, freeze_backbone=False)
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)
model.eval()
print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

explainer = GradCAMExplainer(model, device=DEVICE)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Generate gallery
# ──────────────────────────────────────────────────────────────
total = 0

for folder_str, label, n_samples in SOURCES:
    folder = Path(folder_str)
    if not folder.exists():
        print(f"SKIP {label}: folder not found", flush=True)
        continue

    images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS]
    chosen = random.sample(images, min(n_samples, len(images)))

    print(f"\n[{label}] {len(images)} images, picking {len(chosen)}", flush=True)
    for img_path in chosen:
        out_path = OUTPUT_DIR / f"gradcam_{label}_{img_path.stem}.png"
        if out_path.exists():
            print(f"  skip (exists): {out_path.name}", flush=True)
            continue
        try:
            tensor, rgb_image = load_image_for_gradcam(img_path)
            explainer.save(tensor, rgb_image, out_path)
            print(f"  saved: {out_path.name}", flush=True)
            total += 1
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}", flush=True)

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
all_files = list(OUTPUT_DIR.glob("gradcam_*.png"))
print(f"\nDone! {total} new Grad-CAM images generated.")
print(f"Total gallery: {len(all_files)} images in {OUTPUT_DIR}")
