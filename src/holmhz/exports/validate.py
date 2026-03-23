"""
ONNX Validation — Verify ONNX model matches PyTorch output.

Usage:
    from holmhz.exports.validate import validate_onnx
    max_diff = validate_onnx(pytorch_model, "model.onnx")
    assert max_diff < 1e-5
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def validate_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    tolerance: float = 1e-5,
    n_samples: int = 5,
) -> float:
    """Validate ONNX model output matches PyTorch model.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX file
        input_shape: Input shape for test
        tolerance: Max allowed difference
        n_samples: Number of random samples to test

    Returns:
        Maximum absolute difference across all samples

    Raises:
        AssertionError: If max diff exceeds tolerance
    """
    pytorch_model.eval()
    session = ort.InferenceSession(str(onnx_path))

    max_diff = 0.0

    for _ in range(n_samples):
        test_input = torch.randn(*input_shape)

        # PyTorch output
        with torch.no_grad():
            pt_output = pytorch_model(test_input).numpy()

        # ONNX output
        ort_inputs = {session.get_inputs()[0].name: test_input.numpy()}
        ort_output = session.run(None, ort_inputs)[0]

        diff = np.max(np.abs(pt_output - ort_output))
        max_diff = max(max_diff, diff)

    if max_diff > tolerance:
        raise AssertionError(
            f"ONNX validation failed: max_diff={max_diff:.2e} > tolerance={tolerance:.2e}"
        )

    return max_diff
