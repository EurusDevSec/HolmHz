"""
ONNX Export — Convert PyTorch detector to ONNX format.

Usage:
    from holmhz.exports.onnx_export import export_to_onnx
    export_to_onnx(model, "model.onnx", opset_version=17)
"""

from pathlib import Path

import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    opset_version: int = 17,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    dynamic_axes: dict | None = None,
    simplify: bool = True,
) -> Path:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model (detector)
        output_path: Path to save .onnx file
        opset_version: ONNX opset version
        input_shape: Input tensor shape for tracing
        dynamic_axes: Dynamic axes config (e.g. batch dim)
        simplify: Run onnx-simplifier if available

    Returns:
        Path to exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx.load(str(output_path))
            simplified, ok = onnx_simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(output_path))
        except ImportError:
            pass  # onnxsim not installed — skip simplification

    return output_path
