"""YAML loading and validation for `ModelFamily` definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from edgeadapt.registry.constants import (
    ALLOWED_BACKENDS,
    ALLOWED_POWER,
    ALLOWED_TIERS,
    FAMILY_SCHEMA_VERSION,
)
from edgeadapt.registry.family import ModelFamily, ModelVariant, VariantMetadata


class FamilyYamlError(ValueError):
    """Raised when a family YAML file is invalid."""


def load_family_yaml(path: str | Path) -> ModelFamily:
    """Load and validate a model family from a YAML file."""
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise FamilyYamlError("root YAML document must be a mapping")
    return _parse_family(data, source=str(p))


def _parse_family(data: dict[str, Any], *, source: str) -> ModelFamily:
    name = data.get("family")
    if not isinstance(name, str) or not name.strip():
        raise FamilyYamlError(f"{source}: missing non-empty string field 'family'")

    schema_raw = data.get("schema_version", FAMILY_SCHEMA_VERSION)
    if not isinstance(schema_raw, int):
        raise FamilyYamlError(f"{source}: 'schema_version' must be an integer")
    if schema_raw != FAMILY_SCHEMA_VERSION:
        raise FamilyYamlError(
            f"{source}: unsupported family schema_version {schema_raw}; "
            f"this EdgeAdapt build supports {FAMILY_SCHEMA_VERSION}"
        )

    variants_raw = data.get("variants")
    if not isinstance(variants_raw, dict) or not variants_raw:
        raise FamilyYamlError(f"{source}: 'variants' must be a non-empty mapping")

    variants: dict[str, ModelVariant] = {}
    for key, spec in variants_raw.items():
        if not isinstance(key, str) or not key.strip():
            raise FamilyYamlError(f"{source}: variant keys must be non-empty strings")
        if key in variants:
            raise FamilyYamlError(f"{source}: duplicate variant name {key!r}")
        if not isinstance(spec, dict):
            raise FamilyYamlError(f"{source}: variant {key!r} must be a mapping")
        variants[key] = _parse_variant(key, spec, source=source)

    return ModelFamily(name=name.strip(), variants=variants)


def _parse_variant(name: str, spec: dict[str, Any], *, source: str) -> ModelVariant:
    path = spec.get("path")
    if not isinstance(path, str) or not path.strip():
        raise FamilyYamlError(f"{source}: variant {name!r} missing 'path'")

    tier = spec.get("tier")
    if not isinstance(tier, str) or tier not in ALLOWED_TIERS:
        raise FamilyYamlError(
            f"{source}: variant {name!r} has invalid 'tier' (use high|medium|low)"
        )

    backend = spec.get("backend", "onnx")
    if not isinstance(backend, str) or backend not in ALLOWED_BACKENDS:
        raise FamilyYamlError(f"{source}: variant {name!r} has invalid 'backend'")

    meta_raw = spec.get("metadata") or {}
    if meta_raw is not None and not isinstance(meta_raw, dict):
        raise FamilyYamlError(f"{source}: variant {name!r} 'metadata' must be a mapping")

    metadata = _parse_metadata(name, meta_raw or {}, source=source)

    return ModelVariant(
        name=name,
        path=path.strip(),
        tier=tier,
        backend=backend,
        metadata=metadata,
    )


def _parse_metadata(variant: str, meta: dict[str, Any], *, source: str) -> VariantMetadata:
    lat = meta.get("expected_latency_ms", 0.0)
    mem = meta.get("memory_footprint_mb", 0.0)
    acc = meta.get("accuracy_score", 0.0)
    pwr = meta.get("power_draw_estimate", "medium")

    if not isinstance(lat, (int, float)) or lat < 0:
        raise FamilyYamlError(f"{source}: variant {variant!r} invalid expected_latency_ms")
    if not isinstance(mem, (int, float)) or mem < 0:
        raise FamilyYamlError(f"{source}: variant {variant!r} invalid memory_footprint_mb")
    if not isinstance(acc, (int, float)) or not (0.0 <= float(acc) <= 1.0):
        raise FamilyYamlError(f"{source}: variant {variant!r} invalid accuracy_score")
    if not isinstance(pwr, str) or pwr not in ALLOWED_POWER:
        raise FamilyYamlError(f"{source}: variant {variant!r} invalid power_draw_estimate")

    return VariantMetadata(
        expected_latency_ms=float(lat),
        memory_footprint_mb=float(mem),
        accuracy_score=float(acc),
        power_draw_estimate=pwr,
    )
