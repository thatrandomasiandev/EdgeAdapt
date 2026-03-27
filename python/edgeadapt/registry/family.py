"""Model family and variant types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from edgeadapt.registry.constants import ALLOWED_BACKENDS, ALLOWED_POWER, ALLOWED_TIERS


@dataclass
class VariantMetadata:
    """Benchmarked resource profile for a variant."""

    expected_latency_ms: float
    memory_footprint_mb: float
    accuracy_score: float
    power_draw_estimate: str

    def __post_init__(self) -> None:
        if self.expected_latency_ms < 0 or self.memory_footprint_mb < 0:
            raise ValueError("latency and memory must be non-negative")
        if not 0.0 <= self.accuracy_score <= 1.0:
            raise ValueError("accuracy_score must be between 0 and 1")
        if self.power_draw_estimate not in ALLOWED_POWER:
            raise ValueError("power_draw_estimate must be high|medium|low")


@dataclass
class ModelVariant:
    """A single model artifact within a family."""

    name: str
    path: str
    tier: str
    backend: str = "onnx"
    metadata: VariantMetadata = field(
        default_factory=lambda: VariantMetadata(0.0, 0.0, 0.0, "medium")
    )

    def __post_init__(self) -> None:
        if self.tier not in ALLOWED_TIERS:
            raise ValueError("tier must be one of high|medium|low")
        if self.backend not in ALLOWED_BACKENDS:
            raise ValueError("backend must be one of onnx|tflite|coreml")


@dataclass
class ModelFamily:
    """Named set of model variants."""

    name: str
    variants: dict[str, ModelVariant] = field(default_factory=dict)

    def add_variant(
        self,
        name: str,
        path: str,
        tier: str,
        backend: str = "onnx",
        metadata: VariantMetadata | None = None,
    ) -> None:
        """Register a variant; raises `ValueError` on duplicates or invalid fields."""
        if name in self.variants:
            raise ValueError(f"duplicate variant name: {name!r}")
        meta = metadata or VariantMetadata(0.0, 0.0, 0.0, "medium")
        self.variants[name] = ModelVariant(
            name=name,
            path=path,
            tier=tier,
            backend=backend,
            metadata=meta,
        )

    @classmethod
    def from_yaml(cls, path: str) -> ModelFamily:
        """Load a family from YAML on disk."""
        from edgeadapt.registry.yaml_loader import load_family_yaml

        return load_family_yaml(path)

    def resolve_paths(self, base: str | Path) -> ModelFamily:
        """Return a copy with variant paths resolved relative to `base`."""
        root = Path(base).resolve()
        out: dict[str, ModelVariant] = {}
        for k, v in self.variants.items():
            p = Path(v.path)
            if not p.is_absolute():
                p = (root / p).resolve()
            out[k] = ModelVariant(
                name=v.name,
                path=str(p),
                tier=v.tier,
                backend=v.backend,
                metadata=v.metadata,
            )
        return ModelFamily(name=self.name, variants=out)
