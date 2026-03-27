"""Benchmark runner and reports."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from edgeadapt.backends.onnx_backend import ONNXBackend
from edgeadapt.registry.family import ModelFamily, ModelVariant, VariantMetadata

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _edgeadapt_version() -> str:
    try:
        from importlib.metadata import version

        return version("edgeadapt")
    except Exception:
        return "unknown"


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            timeout=3,
        )
        return out.strip()
    except Exception:
        return None


@dataclass
class VariantBenchRow:
    """Per-variant latency distribution and resource estimate."""

    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    memory_mb: float
    warmup_runs: int
    timed_runs: int
    input_shape: tuple[int, ...]


@dataclass
class BenchmarkReport:
    """Per-variant timing results (v2) plus run metadata."""

    rows: dict[str, VariantBenchRow] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot."""
        variants: dict[str, Any] = {}
        for k, v in self.rows.items():
            d = asdict(v)
            d["input_shape"] = list(v.input_shape)
            variants[k] = d
        return {"variants": variants, "metadata": dict(self.metadata)}


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
    warmup_runs: int = 3,
) -> BenchmarkReport:
    """Run each variant; if ``input_sample`` is None, a matching random tensor is used per variant."""
    meta: dict[str, Any] = {
        "edgeadapt_version": _edgeadapt_version(),
        "onnxruntime_version": ort.__version__,
        "python": sys.version.split()[0],
        "git_commit": _git_commit(),
        "warmup_runs": warmup_runs,
        "timed_runs_per_variant": runs,
    }
    report = BenchmarkReport(metadata=meta)
    input_shapes: dict[str, list[int]] = {}

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
        shape_tuple = tuple(int(x) for x in arr.shape)
        input_shapes[name] = list(shape_tuple)

        for _ in range(warmup_runs):
            _ = backend.infer(arr)

        samples: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = backend.infer(arr)
            samples.append((time.perf_counter() - t0) * 1000.0)

        backend.unload()
        lat = np.asarray(samples, dtype=np.float64)
        mem_mb = float(v.metadata.memory_footprint_mb) if v.metadata.memory_footprint_mb else 0.0
        report.rows[name] = VariantBenchRow(
            latency_ms_mean=float(np.mean(lat)),
            latency_ms_p50=float(np.percentile(lat, 50)),
            latency_ms_p95=float(np.percentile(lat, 95)),
            latency_ms_p99=float(np.percentile(lat, 99)),
            memory_mb=mem_mb,
            warmup_runs=warmup_runs,
            timed_runs=runs,
            input_shape=shape_tuple,
        )

    meta["input_shapes"] = input_shapes
    return report


def apply_report_to_family(family: ModelFamily, report: BenchmarkReport) -> ModelFamily:
    """Return a copy of ``family`` with metadata updated from ``report``."""
    out: dict[str, ModelVariant] = {}
    for key, v in family.variants.items():
        row = report.rows.get(key)
        if row is None:
            lat = v.metadata.expected_latency_ms
            mem = v.metadata.memory_footprint_mb
        else:
            lat = row.latency_ms_mean
            mem = row.memory_mb
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
