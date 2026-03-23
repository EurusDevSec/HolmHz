"""
Grad-CAM CLI — Tạo heatmap giải thích model nhìn vùng nào.

Usage:
    python scripts/explain.py --image imgs/test.png --model efficientnet_b0 --checkpoint weights/best_model.pt
    python scripts/explain.py --image-dir imgs/Fake_AI_generated/ --model resnet18 --checkpoint weights/best_resnet18.pt --output outputs/xai_gallery/
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors  # noqa: F401 — trigger registry
from holmhz.xai.gradcam import GradCAMExplainer
from holmhz.xai.utils import load_image_for_gradcam


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Explainability")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image-dir", type=str, help="Directory of images")
    parser.add_argument("--model", type=str, required=True, help="Model name in registry")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="outputs/xai_gallery/", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = DETECTOR_REGISTRY.build(args.model, pretrained=False, freeze_backbone=False)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    explainer = GradCAMExplainer(model, device=args.device)

    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths.extend(
            p for p in sorted(Path(args.image_dir).iterdir())
            if p.suffix.lower() in exts
        )

    if not image_paths:
        print("No images found. Use --image or --image-dir.")
        return

    for img_path in image_paths:
        tensor, rgb_image = load_image_for_gradcam(img_path)
        out_path = output_dir / f"gradcam_{img_path.stem}.png"
        explainer.save(tensor, rgb_image, out_path)
        print(f"Saved: {out_path}")

    print(f"Done. {len(image_paths)} images processed → {output_dir}")


if __name__ == "__main__":
    main()
