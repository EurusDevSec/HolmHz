# scripts/benchmark_sota.py
"""
Benchmark SOTA models trên HolmHz test set.

Dùng repos + weights đã có sẵn từ Phase 0:
  prac/ai-experiments/deepfake-detection/
  ├── CNNDetection/          (weights/blur_jpg_prob0.5.pth)
  ├── UniversalFakeDetect/   (pretrained_weights/fc_weights.pth)
  └── DeepfakeBench/         (training/pretrained/effnb4_best.pth)

Usage:
    python scripts/benchmark_sota.py --model holmhz
    python scripts/benchmark_sota.py --model cnndetection
    python scripts/benchmark_sota.py --model universalfake
    python scripts/benchmark_sota.py --model deepfakebench

Output:
    outputs/benchmark/predictions/{model_name}_predictions.csv
"""

import argparse
import csv
import json
import sys
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Base path to Phase 0 repos
PRAC_BASE = Path("prac/ai-experiments/deepfake-detection")


def load_manifest(manifest_path: str) -> list[dict]:
    """Load manifest JSON → list of {path, label, source}."""
    with open(manifest_path) as f:
        return json.load(f)


# ============================================================
# MODEL RUNNERS — mỗi model có preprocessing riêng
# ============================================================

def run_holmhz(samples: list[dict], device: str) -> list[float]:
    """HolmHz v4 — EfficientNet-B0, ImageNet norm, 224x224."""
    import holmhz.detectors  # noqa: F401 — registers detectors
    from holmhz.utils.registry import DETECTOR_REGISTRY

    model = DETECTOR_REGISTRY.build(
        "efficientnet_b0", pretrained=False, dropout=0.3, freeze_backbone=False,
    )
    ckpt = torch.load(
        "outputs/checkpoints/best_v4.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    probs = []
    for sample in tqdm(samples, desc="HolmHz v4"):
        img = Image.open(sample["path"]).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(x)
            prob = torch.sigmoid(logit).item()
        probs.append(prob)
    return probs


def run_cnndetection(samples: list[dict], device: str) -> list[float]:
    """CNNDetection — ResNet-50, ImageNet norm, NO resize (original size).

    ⚠️ CNNDetection demo.py does NOT resize images.
    It uses ToTensor + ImageNet Normalize on original resolution.
    We keep this behavior for fair comparison.
    """
    repo_path = PRAC_BASE / "CNNDetection"
    sys.path.insert(0, str(repo_path))
    from networks.resnet import resnet50

    model = resnet50(num_classes=1)
    state = torch.load(
        str(repo_path / "weights" / "blur_jpg_prob0.5.pth"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    # ⚠️ Theo demo.py gốc: KHÔNG resize, chỉ ToTensor + Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    probs = []
    for sample in tqdm(samples, desc="CNNDetection"):
        img = Image.open(sample["path"]).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(x).sigmoid().item()
        probs.append(prob)
    return probs


def run_universalfake(samples: list[dict], device: str) -> list[float]:
    """UniversalFakeDetect — CLIP ViT-L/14 + Linear, CLIP preprocessing.

    ⚠️ PHẢI dùng CLIP preprocess (khác ImageNet!).
    Theo test_universal.py: model.preprocess từ CLIP load.
    """
    repo_path = PRAC_BASE / "UniversalFakeDetect"
    sys.path.insert(0, str(repo_path))
    from models import get_model

    model = get_model("CLIP:ViT-L/14")
    state_dict = torch.load(
        str(repo_path / "pretrained_weights" / "fc_weights.pth"),
        map_location="cpu",
        weights_only=False,
    )
    model.fc.load_state_dict(state_dict)
    model = model.to(device).eval()

    # CLIP preprocessing — KHÔNG dùng ImageNet norm
    transform = model.preprocess

    probs = []
    for sample in tqdm(samples, desc="UniversalFakeDetect"):
        img = Image.open(sample["path"]).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(x).sigmoid().item()
        probs.append(prob)
    return probs


def run_deepfakebench(samples: list[dict], device: str) -> list[float]:
    """DeepfakeBench — EfficientNet-B4, [0.5,0.5,0.5] norm, 256x256.

    ⚠️ Cần mock tensorboard + dlib + các dependencies nặng.
    Theo test_deepfakebench.py: resize 256, norm [0.5,0.5,0.5].
    """
    import yaml
    import importlib.util
    import types
    from unittest.mock import MagicMock

    repo_path = PRAC_BASE / "DeepfakeBench"
    training_path = repo_path / "training"

    # Add training path to sys.path
    if str(training_path) not in sys.path:
        sys.path.insert(0, str(training_path))

    # Mock tensorboard
    sys.modules.setdefault('torch.utils.tensorboard', MagicMock())

    # ── Step 1: Load registry (simple, no heavy deps) ──
    import metrics.registry as registry_mod
    BACKBONE = registry_mod.BACKBONE
    DETECTOR = registry_mod.DETECTOR
    LOSSFUNC = registry_mod.LOSSFUNC

    # ── Step 2: Load EfficientNetB4 backbone (needs efficientnet_pytorch) ──
    from networks.efficientnetb4 import EfficientNetB4

    # ── Step 3: Register dummy cross_entropy loss ──
    import torch.nn as _nn

    @LOSSFUNC.register_module(module_name='cross_entropy')
    class _DummyCE(_nn.Module):
        def forward(self, pred, label):
            return torch.nn.functional.cross_entropy(pred, label)

    # ── Step 4: Load base_detector.py via importlib (bypass detectors/__init__) ──
    def _load_module(name, filepath):
        spec = importlib.util.spec_from_file_location(name, str(filepath))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    base_det_mod = _load_module(
        "detectors.base_detector",
        training_path / "detectors" / "base_detector.py",
    )
    AbstractDetector = base_det_mod.AbstractDetector

    # ── Step 5: Prepare fake 'loss' and 'detectors' packages in sys.modules ──
    # so that `from loss import LOSSFUNC` and `from detectors import DETECTOR`
    # inside efficientnetb4_detector.py resolve correctly.
    fake_loss = types.ModuleType('loss')
    fake_loss.LOSSFUNC = LOSSFUNC
    sys.modules['loss'] = fake_loss

    fake_detectors = types.ModuleType('detectors')
    fake_detectors.DETECTOR = DETECTOR
    fake_detectors.base_detector = base_det_mod
    sys.modules['detectors'] = fake_detectors
    sys.modules['detectors.base_detector'] = base_det_mod

    fake_networks = types.ModuleType('networks')
    fake_networks.BACKBONE = BACKBONE
    sys.modules['networks'] = fake_networks

    # ── Step 6: Load efficientnetb4_detector.py ──
    efb4_mod = _load_module(
        "detectors.efficientnetb4_detector",
        training_path / "detectors" / "efficientnetb4_detector.py",
    )
    EfficientDetector = efb4_mod.EfficientDetector

    # Load config
    conf_path = training_path / "config" / "detector" / "efficientnetb4.yaml"
    with open(conf_path) as f:
        config = yaml.safe_load(f)
    config['pretrained'] = None
    config.setdefault('backbone_config', {
        'num_classes': 2, 'inc': 3, 'dropout': False, 'mode': 'Original'
    })
    config.setdefault('loss_func', 'cross_entropy')

    # Init model + load weights
    model = EfficientDetector(config=config)
    ckpt = torch.load(
        str(training_path / "pretrained" / "effnb4_best.pth"),
        map_location="cpu",
        weights_only=False,
    )
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # DeepfakeBench: resize 256, norm [0.5, 0.5, 0.5]
    resolution = config.get('resolution', 256)
    mean = config.get('mean', [0.5, 0.5, 0.5])
    std = config.get('std', [0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    probs = []
    for sample in tqdm(samples, desc="DeepfakeBench"):
        img = Image.open(sample["path"]).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            data_dict = {'image': x, 'label': None}
            pred_dict = model(data_dict, inference=True)
            prob = pred_dict['prob'].item()
        probs.append(prob)
    return probs


# ============================================================
MODEL_RUNNERS = {
    "holmhz": run_holmhz,
    "cnndetection": run_cnndetection,
    "universalfake": run_universalfake,
    "deepfakebench": run_deepfakebench,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark SOTA models on HolmHz test set")
    parser.add_argument("--model", required=True, choices=list(MODEL_RUNNERS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load test sets
    id_samples = load_manifest("data/manifests/test_id.json")
    ood_samples = load_manifest("data/manifests/test_ood.json")
    all_samples = id_samples + ood_samples

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {args.model}")
    print(f"  Samples: {len(all_samples)} (ID: {len(id_samples)}, OOD: {len(ood_samples)})")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    # Run inference
    runner = MODEL_RUNNERS[args.model]
    probs = runner(all_samples, args.device)

    # Save predictions CSV
    output_dir = Path("outputs/benchmark/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}_predictions.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "source", "split", "prob_fake"])
        for i, sample in enumerate(all_samples):
            split = "id" if i < len(id_samples) else "ood"
            writer.writerow([
                sample["path"],
                sample["label"],
                sample["source"],
                split,
                f"{probs[i]:.6f}",
            ])

    print(f"\n✅ Predictions saved: {output_path}")
    print(f"   Total: {len(probs)} samples")


if __name__ == "__main__":
    main()