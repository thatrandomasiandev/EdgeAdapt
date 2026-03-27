"""Benchmark runner and reports."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnxruntime as ort

from edgeadapt.backends.onnx_backend import ONNXBackend
from edgeadapt.registry.family import ModelFamily, ModelVariant, VariantMetadata


@dataclass
class BenchmarkReport:
    """Per-variant timing results."""

    rows: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_row(self, variant: str, latency_ms: float, memory_mb: float) -> None:
        """Record metrics for a variant."""
        self.rows[variant] = {"latency_ms": latency_ms, "memory_mb": memory_mb}


def _sample_input_for_model(path: str) -> np.ndarray:
    """Build a random float32 tensor matching the model's first input shape."""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape
    fixed: list[int] = []
    for dim in shape:
        if isinstance(dim, str) or dim is None:
            fixed.append(1)
        else:
            fixed.append(int(dim))
    return np.random.randn(*fixed).astype(np.float32)


def benchmark_family(
    family: ModelFamily,
    input_sample: Any | None = None,
    *,
    runs: int = 100,
) -> BenchmarkReport:
    """Run each variant; if ``input_sample`` is None, a matching random tensor is used."""
    report = BenchmarkReport()
    for name, v in family.variants.items():
        backend = ONNXBackend()
        backend.load(v.path)
        arr = (
            np.asarray(input_sample)
            if input_sample is not None
            else _sample_input_for_model(v.path)
        )
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        # Warmup
        for _ in range(3):
            _ = backend.infer(arr)
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = backend.infer(arr)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) / max(1, runs) * 1000.0
        backend.unload()
        mem_mb = float(v.metadata.memory_footprint_mb) if v.metadata.memory_footprint_mb else 0.0
        report.add_row(name, latency_ms=latency_ms, memory_mb=mem_mb)
    return report


def apply_report_to_family(family: ModelFamily, report: BenchmarkReport) -> ModelFamily:
    """Return a copy of ``family`` with metadata updated from ``report``."""
    out: dict[str, ModelVariant] = {}
    for key, v in family.variants.items():
        row = report.rows.get(key, {})
        lat = float(row.get("latency_ms", v.metadata.expected_latency_ms))
        mem = float(row.get("memory_mb", v.metadata.memory_footprint_mb))
        meta = VariantMetadata(
            expected_latency_ms=lat,
            memory_footprint_mb=mem,
            accuracy_score=v.metadata.accuracy_score,
            power_draw_estimate=v.metadata.power_draw_estimate,
        )
        out[key] = ModelVariant(
            name=v.name,
            path=v.path,
            tier=v.tier,
            backend=v.backend,
            metadata=meta,
        )
    return ModelFamily(name=family.name, variants=out)
