"""ONNX backend tests."""

from __future__ import annotations

import numpy as np
import pytest

from edgeadapt.backends.onnx_backend import ONNXBackend


def test_onnx_load_infer_unload(dummy_model_paths: dict[str, object]) -> None:
    """Load, infer, unload cycle."""
    path = str(dummy_model_paths["low"])
    b = ONNXBackend()
    b.load(path)
    x = np.random.randn(1, 4).astype(np.float32)
    y = b.infer(x)
    assert np.asarray(y).shape[0] == 1
    b.unload()
    assert b.is_loaded() is False


def test_bad_path_raises() -> None:
    """Invalid path raises."""
    b = ONNXBackend()
    with pytest.raises(OSError):
        b.load("/nonexistent/model.onnx")


def test_capabilities_after_load(dummy_model_paths: dict[str, object]) -> None:
    """ONNX backend reports capability descriptor."""
    path = str(dummy_model_paths["low"])
    b = ONNXBackend()
    b.load(path)
    cap = b.capabilities()
    assert cap.backend_id == "onnxruntime"
    assert cap.supports_concurrent_infer is True
    assert len(cap.input_names) >= 1
    assert b.last_load_time_sec is not None and b.last_load_time_sec >= 0.0
