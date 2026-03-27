"""Tests for model registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgeadapt.registry.family import ModelFamily, ModelVariant, VariantMetadata
from edgeadapt.registry.yaml_loader import FamilyYamlError


def test_add_variant_duplicate() -> None:
    """Duplicate variant names raise."""
    fam = ModelFamily(name="t")
    fam.add_variant("a", "p", "high")
    with pytest.raises(ValueError, match="duplicate"):
        fam.add_variant("a", "p2", "low")


def test_invalid_tier() -> None:
    """Invalid tier is rejected."""
    with pytest.raises(ValueError):
        ModelVariant(name="a", path="p", tier="invalid", metadata=VariantMetadata(1, 1, 0.5, "low"))


def test_yaml_roundtrip(tmp_path: Path) -> None:
    """Load a minimal YAML family."""
    p = tmp_path / "f.yaml"
    p.write_text(
        """
family: test_family
variants:
  v1:
    path: models/m.onnx
    tier: high
    backend: onnx
    metadata:
      expected_latency_ms: 10
      memory_footprint_mb: 20
      accuracy_score: 0.9
      power_draw_estimate: high
""",
        encoding="utf-8",
    )
    fam = ModelFamily.from_yaml(str(p))
    assert fam.name == "test_family"
    assert "v1" in fam.variants


def test_yaml_invalid_root(tmp_path: Path) -> None:
    """Non-mapping YAML root raises."""
    p = tmp_path / "bad.yaml"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(FamilyYamlError):
        ModelFamily.from_yaml(str(p))


def test_yaml_unsupported_schema_version(tmp_path: Path) -> None:
    """Unsupported schema_version raises."""
    p = tmp_path / "bad_schema.yaml"
    p.write_text(
        """
schema_version: 999
family: x
variants:
  a:
    path: m.onnx
    tier: high
    backend: onnx
    metadata:
      expected_latency_ms: 1
      memory_footprint_mb: 1
      accuracy_score: 0.5
      power_draw_estimate: low
""",
        encoding="utf-8",
    )
    with pytest.raises(FamilyYamlError, match="unsupported family schema_version"):
        ModelFamily.from_yaml(str(p))
