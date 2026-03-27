"""Pytest fixtures for EdgeAdapt."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_MODELS = ROOT / "benchmarks" / "sample_models"


@pytest.fixture
def dummy_model_paths() -> dict[str, Path]:
    """Paths to generated dummy ONNX models."""
    return {
        "high": SAMPLE_MODELS / "dummy_high.onnx",
        "medium": SAMPLE_MODELS / "dummy_medium.onnx",
        "low": SAMPLE_MODELS / "dummy_low.onnx",
    }
