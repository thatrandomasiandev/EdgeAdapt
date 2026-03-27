"""Benchmark runner v2 report shape."""

from __future__ import annotations

import json

from edgeadapt.benchmark.runner import benchmark_family
from edgeadapt.registry.family import ModelFamily, VariantMetadata


def test_benchmark_report_v2_shape(dummy_model_paths: dict[str, object]) -> None:
    """Report includes percentiles and metadata."""
    fam = ModelFamily(name="t")
    fam.add_variant(
        "low",
        str(dummy_model_paths["low"]),
        "low",
        metadata=VariantMetadata(1.0, 1.0, 0.5, "low"),
    )
    report = benchmark_family(fam, runs=10, warmup_runs=2)
    row = report.rows["low"]
    assert row.timed_runs == 10
    assert row.warmup_runs == 2
    assert row.latency_ms_p50 <= row.latency_ms_p95 <= row.latency_ms_p99 + 1e-6
    assert "onnxruntime_version" in report.metadata
    assert "input_shapes" in report.metadata
    raw = json.dumps(report.to_dict())
    assert "low" in raw
