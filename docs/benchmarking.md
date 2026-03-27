# Benchmarking methodology

EdgeAdapt benchmarks each variant in a `ModelFamily` independently using ONNX Runtime on CPU.

## What is measured

- **Per-infer latency:** After a configurable **warmup** phase, the runner records one wall-clock sample per inference for `timed_runs` iterations. It reports **mean**, **p50**, **p95**, and **p99** (milliseconds) over those samples.
- **Memory:** The value stored is the variant’s **declared** `memory_footprint_mb` from metadata (not a live RSS snapshot), consistent with updating family metadata after a benchmark run.
- **Input shape:** Each variant uses either a provided `--sample` numpy array or a random tensor matching that model’s first ONNX input shape (dynamic dimensions are fixed to `1` for sampling).

## BenchmarkReport v2 metadata

`BenchmarkReport` includes:

| Field | Meaning |
|-------|---------|
| `edgeadapt_version` | Installed Python package version (or `unknown`). |
| `onnxruntime_version` | ONNX Runtime version string. |
| `python` | Short Python version. |
| `git_commit` | `git rev-parse HEAD` when run inside a git checkout (else `null`). |
| `warmup_runs` / `timed_runs_per_variant` | Warmup and timed iteration counts. |
| `input_shapes` | Map of variant name → list of input dimensions used. |

## CLI

```bash
edgeadapt benchmark --config path/to/family.yaml --runs 100
edgeadapt benchmark --config path/to/family.yaml --runs 100 --json
```

Use `--json` for machine-readable output suitable for CI artifacts or comparisons.

## Comparability

- Use the **same CPU**, **same ORT version**, and **same runs** when comparing reports.
- Prefer a fixed `--sample` (`.npy`) when inputs should be identical across runs.
- p95/p99 are sensitive to **background load**; run on a quiet machine for stable tail latency.
