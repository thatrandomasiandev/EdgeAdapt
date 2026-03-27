"""Inspect ONNX models via ONNX Runtime (IO metadata, providers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnxruntime as ort


def _shape_to_json(shape: object) -> list[Any] | str:
    """Convert a session shape (may contain strings for dynamic dims) to JSON-friendly form."""
    if shape is None:
        return []
    if isinstance(shape, (list, tuple)):
        out: list[Any] = []
        for dim in shape:
            if isinstance(dim, str):
                out.append(dim)
            else:
                out.append(int(dim))
        return out
    return str(shape)


def describe_onnx_model(path: str | Path) -> dict[str, Any]:
    """Load ``path`` with ONNX Runtime and return inputs, outputs, and provider info.

    This does **not** validate numerical correctness; it surfaces graph IO metadata
    for debugging and tooling.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"model path does not exist: {p}")

    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    inputs = [
        {
            "name": i.name,
            "shape": _shape_to_json(i.shape),
            "type": str(i.type),
        }
        for i in sess.get_inputs()
    ]
    outputs = [
        {
            "name": o.name,
            "shape": _shape_to_json(o.shape),
            "type": str(o.type),
        }
        for o in sess.get_outputs()
    ]
    out: dict[str, Any] = {
        "path": str(p.resolve()),
        "inputs": inputs,
        "outputs": outputs,
        "providers_available": sess.get_providers(),
    }
    try:
        out["provider_options"] = sess.get_provider_options()
    except (AttributeError, TypeError, RuntimeError):
        out["provider_options"] = None
    return out
