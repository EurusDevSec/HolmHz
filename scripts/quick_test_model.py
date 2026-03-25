"""
Quick test: HolmHz ONNX model on user's sample images.
Shows predictions for all imgs/ images to validate current model behavior.
"""
import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():
    import onnxruntime as ort

    PROJECT = Path(__file__).parent.parent
    ONNX_PATH = PROJECT / "outputs" / "exports" / "efficientnet_b0.onnx"
    THRESHOLD = 0.76

    # ImageNet norm
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def predict(img_path):
        img = Image.open(img_path).resize((224, 224)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis]
        arr = (arr - mean) / std
        logit = session.run(None, {input_name: arr.astype(np.float32)})[0][0][0]
        prob = float(1.0 / (1.0 + np.exp(-float(logit))))
        label = "FAKE" if prob >= THRESHOLD else "REAL"
        return prob, label

    # Scan all images
    img_dir = PROJECT / "imgs"
    print("=" * 70)
    print("HolmHz v4 (EfficientNet-B0) — Sample Image Test")
    print(f"Threshold: {THRESHOLD}")
    print("=" * 70)

    correct = 0
    total = 0

    for category in ["Fake_AI_generated", "Real"]:
        cat_dir = img_dir / category
        true_label = "FAKE" if "Fake" in category else "REAL"
        print(f"\n--- {category} (Ground Truth: {true_label}) ---")

        for img_file in sorted(cat_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                prob, pred = predict(img_file)
                ok = "OK" if pred == true_label else "WRONG"
                if pred == true_label:
                    correct += 1
                total += 1
                print(f"  {img_file.name[:45]:45s}  P(fake)={prob:.4f}  -> {pred:4s}  [{ok}]")

    print(f"\n{'=' * 70}")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
