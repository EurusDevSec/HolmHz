"""
HolmHz Prediction Script — Inference trên 1 hoặc nhiều ảnh.

Usage:
    # Predict 1 ảnh
    python scripts/predict.py imgs/Real/IMG_2344.jpg

    # Predict cả folder
    python scripts/predict.py imgs/Fake_AI_generated/

    # Custom checkpoint
    python scripts/predict.py imgs/Real/IMG_2344.jpg --checkpoint outputs/checkpoints/best.pt
"""

import sys
from pathlib import Path

import cv2
import torch
from dotenv import load_dotenv

load_dotenv()

import holmhz.detectors  # noqa: E402, F401
from holmhz.data.transforms import get_val_transforms
from holmhz.utils.logger import get_logger
from holmhz.utils.registry import DETECTOR_REGISTRY

logger = get_logger("predict")


def predict_single(
    image_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> dict:
    """Predict 1 ảnh.

    Returns:
        dict: {"path", "prob_fake", "label", "confidence"}
    """
    # Load ảnh
    img = cv2.imread(image_path)
    if img is None:
        return {"path": image_path, "error": "Cannot load image"}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform
    transformed = transform(image=img)
    tensor = transformed["image"].unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor)                    # [1, 1]
        prob = torch.sigmoid(logits).item()       # P(Fake) ∈ [0, 1]

    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else 1 - prob

    return {
        "path": image_path,
        "prob_fake": round(prob, 4),
        "label": label,
        "confidence": round(confidence, 4),
    }


def main():
    """Main prediction entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <image_or_folder> [--checkpoint path]")
        sys.exit(1)

    input_path = sys.argv[1]
    checkpoint_path = "outputs/checkpoints/best.pt"

    # Parse --checkpoint arg
    if "--checkpoint" in sys.argv:
        idx = sys.argv.index("--checkpoint")
        checkpoint_path = sys.argv[idx + 1]

    # Check checkpoint exists
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Train a model first: python scripts/train.py")
        sys.exit(1)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model = DETECTOR_REGISTRY.build(
        "efficientnet_b0",
        pretrained=False,   # Sẽ load weights từ checkpoint
        dropout=0.3,
        freeze_backbone=False,
    )

    # Load checkpoint (chỉ model weights)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    best_auc = checkpoint.get("best_metric", "N/A")
    epoch = checkpoint.get("epoch", "N/A")
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch}, AUC {best_auc})")

    # Transform (val transform — no augmentation)
    transform = get_val_transforms(image_size=224)

    # Collect image paths
    input_p = Path(input_path)
    if input_p.is_file():
        image_paths = [str(input_p)]
    elif input_p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            str(f) for f in input_p.rglob("*") if f.suffix.lower() in exts
        )
    else:
        logger.error(f"Path not found: {input_path}")
        sys.exit(1)

    if not image_paths:
        logger.error(f"No images found in: {input_path}")
        sys.exit(1)

    logger.info(f"Predicting {len(image_paths)} images...")

    # Predict
    results = []
    print("\n" + "=" * 70)
    print(f"{'Path':<50} {'Label':<6} {'P(Fake)':<8} {'Conf':<6}")
    print("=" * 70)

    for img_path in image_paths:
        result = predict_single(img_path, model, transform, device)
        results.append(result)

        if "error" in result:
            print(f"{Path(img_path).name:<50} ERROR: {result['error']}")
        else:
            name = Path(img_path).name
            if len(name) > 48:
                name = name[:45] + "..."
            print(
                f"{name:<50} "
                f"{result['label']:<6} "
                f"{result['prob_fake']:<8.4f} "
                f"{result['confidence']:<6.4f}"
            )

    print("=" * 70)

    # Summary
    valid = [r for r in results if "error" not in r]
    fakes = sum(1 for r in valid if r["label"] == "FAKE")
    reals = sum(1 for r in valid if r["label"] == "REAL")
    print(f"\nSummary: {len(valid)} images — {fakes} FAKE, {reals} REAL")


if __name__ == "__main__":
    main()
