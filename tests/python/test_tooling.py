"""Tooling and diagnostics tests."""

from __future__ import annotations

from pathlib import Path

from edgeadapt.tooling.environment import environment_report
from edgeadapt.tooling.onnx_inspect import describe_onnx_model


def test_environment_report_keys() -> None:
    """Environment report contains expected keys."""
    rep = environment_report()
    assert "edgeadapt_version" in rep
    assert "onnxruntime_version" in rep
    assert "rust_extension_loaded" in rep


def test_describe_onnx_model(dummy_model_paths: dict[str, object]) -> None:
    """Inspect reports inputs and outputs."""
    path = Path(dummy_model_paths["low"])
    info = describe_onnx_model(path)
    assert info["inputs"]
    assert info["outputs"]
    assert "CPUExecutionProvider" in info["providers_available"]
