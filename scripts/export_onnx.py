"""
ONNX Export CLI — Export trained model and run validation.

Usage:
    python scripts/export_onnx.py configs/export.yaml
    python scripts/export_onnx.py configs/export.yaml --model resnet18 --checkpoint weights/best_resnet18.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from holmhz.utils.registry import DETECTOR_REGISTRY
import holmhz.detectors  # noqa: F401
from holmhz.exports.onnx_export import export_to_onnx
from holmhz.exports.validate import validate_onnx


def benchmark_latency(onnx_path: str, input_shape: tuple, n_runs: int = 100) -> dict:
    """Measure ONNX inference latency on CPU."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_input = np.random.randn(*input_shape).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: test_input})

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: test_input})
        times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
    }


def main():
    parser = argparse.ArgumentParser(description="ONNX Export + Validation")
    parser.add_argument("config", type=str, help="Export config YAML")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--checkpoint", type=str, help="Override checkpoint path")
    parser.add_argument("--benchmark", action="store_true", help="Run CPU latency benchmark")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    model_name = args.model or cfg.model.name
    checkpoint_path = args.checkpoint or cfg.model.checkpoint
    output_dir = Path(cfg.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"

    # Build and load model
    model = DETECTOR_REGISTRY.build(model_name, pretrained=False, freeze_backbone=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model: {model_name}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")

    # Export
    input_shape = tuple(cfg.export.input_shape)
    export_to_onnx(
        model,
        onnx_path,
        opset_version=cfg.export.opset_version,
        input_shape=input_shape,
        simplify=cfg.export.get("simplify", True),
    )
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported: {onnx_path} ({file_size_mb:.1f} MB)")

    # Validate
    if cfg.validation.get("enabled", True):
        tolerance = cfg.validation.tolerance
        max_diff = validate_onnx(model, onnx_path, input_shape=input_shape, tolerance=tolerance)
        print(f"Validation: PASSED (max_diff={max_diff:.2e}, tolerance={tolerance:.0e})")

    # Benchmark
    if args.benchmark:
        print("\nCPU Latency Benchmark (100 runs):")
        stats = benchmark_latency(str(onnx_path), input_shape)
        print(f"  Mean:  {stats['mean_ms']:.2f} ms")
        print(f"  Std:   {stats['std_ms']:.2f} ms")
        print(f"  P50:   {stats['p50_ms']:.2f} ms")
        print(f"  P95:   {stats['p95_ms']:.2f} ms")


if __name__ == "__main__":
    main()
