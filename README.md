# EdgeAdapt

[![CI](https://github.com/thatrandomasiandev/EdgeAdapt/actions/workflows/ci.yml/badge.svg)](https://github.com/thatrandomasiandev/EdgeAdapt/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**EdgeAdapt** is an open-source **adaptive inference control plane** for edge and device-aware deployment. It sits *above* concrete runtimes (today: primarily [ONNX Runtime](https://onnxruntime.ai/)): it **profiles** the device, **selects** among registered model variants using a **policy**, and **hot-swaps** the active model in the background while keeping the previous model serving if a load fails.

This is **not** a tensor engine, an OpenCV-style vision library, or a broad multi-format runtime. The goal is **safe, explainable orchestration**: selection vs activation stay separate—the policy *chooses* variant names; the swapper *activates* backends; the `Engine` *orchestrates*.

**Repository:** [github.com/thatrandomasiandev/EdgeAdapt](https://github.com/thatrandomasiandev/EdgeAdapt)

---

## Table of contents

- [Why EdgeAdapt](#why-edgeadapt)
- [What ships today](#what-ships-today)
- [Architecture at a glance](#architecture-at-a-glance)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Core concepts](#core-concepts)
- [CLI reference](#cli-reference)
- [Production-oriented features](#production-oriented-features)
- [Configuration and extension points](#configuration-and-extension-points)
- [Benchmarking](#benchmarking)
- [Documentation](#documentation)
- [Development](#development)
- [Security](#security)
- [License](#license)

---

## Why EdgeAdapt

On constrained devices, one model size rarely fits all sessions. You may want a smaller model when memory is tight, a larger one when power and latency budgets allow, or a tier that matches a QoS target. EdgeAdapt gives you:

- A **registry** (`ModelFamily` / YAML) for named variants and metadata.
- A **profiler** that turns OS signals into a `DeviceState`.
- **Policies** (built-in presets + `LambdaPolicy`) plus a **StabilityGuard** to reduce thrashing.
- A **Rust-backed hot swapper** that loads in a background thread and only commits on success—**last-known-good** behavior on failure.
- An **`Engine`** API that ties it together with `infer()`, optional observability, and pluggable activation checks.

---

## What ships today

| Area | Status |
|------|--------|
| Device profiling (`DeviceState`) | Implemented (platform-specific details vary; some Linux paths may skip off-target OSes). |
| YAML model families + schema version | Implemented |
| Policies + stability guard | Implemented |
| ONNX backend + `InferenceBackend` ABC | Implemented |
| `BackendCapability` descriptors + ONNX reporting | Implemented |
| Engine: swap safety semantics, reason codes, optional custom `loader` | Implemented |
| Observability: events, counters, `select_with_explanation` | Implemented |
| Benchmarks: `BenchmarkReport` v2 (p50/p95/p99, metadata, JSON export) | Implemented |
| CLI: validate, benchmark, profile, doctor, inspect-model, decision traces | Implemented |
| Second shipping backend (TFLite, Core ML, …) | Roadmap—not required for the core story |

See [`docs/gap_analysis.md`](docs/gap_analysis.md) and [`docs/roadmap.md`](docs/roadmap.md) for deliberate scope and future work.

---

## Architecture at a glance

1. **Profiler** — samples battery, thermals, memory, CPU (and related signals) into `DeviceState`.
2. **Registry** — names variants, paths, tiers, backend tags, and benchmark metadata.
3. **Policy** — maps `(DeviceState, ModelFamily)` → variant name; can expose **`select_with_explanation`** for debugging.
4. **StabilityGuard** — hysteresis, cooldown, frequency limits (optional wrapper around a `BasePolicy`).
5. **Hot swapper** — Python callable loads an `InferenceBackend`; Rust coordinates background load and atomic pointer swap; failures invoke **`on_fallback`** without discarding the previous backend.

The **`Engine`** runs a polling loop: read state → evaluate policy → optionally swap → serve `infer()` on the active backend.

More detail: [`docs/architecture.md`](docs/architecture.md).

---

## Repository layout

| Path | Role |
|------|------|
| `python/edgeadapt/` | Python package: engine, policies, registry, backends, CLI, benchmark runner, observability |
| `rust/edgeadapt_core/` | PyO3 extension: `HotSwapper` and related native glue |
| `tests/python/` | Pytest suite (integration, swap matrix, CLI, benchmarks, contracts) |
| `docs/` | MkDocs sources (architecture, platform, roadmap, gap analysis, swap lifecycle, benchmarking) |
| `examples/` | Example configs and usage patterns |
| `benchmarks/` | Sample assets used in tests/docs |

---

## Requirements

- **Python** 3.10+
- **Rust** toolchain (for building the extension via Maturin)
- **ONNX Runtime** (Python dependency; CPU EP used in typical paths)

---

## Installation

From a clone of [the repository](https://github.com/thatrandomasiandev/EdgeAdapt):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --extras dev
python -c "import edgeadapt; print(edgeadapt.__version__)"
```

This builds and installs the `edgeadapt` package in editable mode with the `edgeadapt` CLI on your `PATH` (depending on your environment).

---

## Quickstart

Minimal pattern: define or load a `ModelFamily`, choose a policy, construct `Engine`, call `start()`, run `infer()`.

```python
from edgeadapt.engine import Engine
from edgeadapt.policy.presets import MaximizeAccuracy
from edgeadapt.registry.family import ModelFamily, VariantMetadata

# ModelFamily is usually loaded from YAML via ModelFamily.from_yaml(...)
family = ModelFamily(name="demo")
family.add_variant(
    "low",
    "/path/to/model.onnx",
    "low",
    metadata=VariantMetadata(
        expected_latency_ms=20.0,
        memory_footprint_mb=50.0,
        accuracy_score=0.85,
        power_draw_estimate="low",
    ),
)

engine = Engine(
    family,
    MaximizeAccuracy(latency_ceiling_ms=100.0),
    poll_interval_sec=5.0,
    initial_variant="low",
)
engine.start()
try:
    output = engine.infer(your_input)  # numpy/tensor matching the model
finally:
    engine.stop()
```

Context-manager usage: `with Engine(...) as engine:` calls `start()` / `stop()` around the block.

---

## Core concepts

### `ModelFamily` and variants

A family groups **variants** (e.g. `low` / `medium` / `high`), each with a path, **tier**, **backend** tag (`onnx`, `tflite`, `coreml` in metadata), and `VariantMetadata` (latency, memory, accuracy, power hints). YAML loading is validated (`schema_version`, required fields). See [`docs/policies.md`](docs/policies.md) and the registry code under `python/edgeadapt/registry/`.

### Policies

- **`BasePolicy`** — `select(state, family) -> str`; optional **`select_with_explanation`** returns `(name, explanation_dict)`.
- **Presets** — e.g. `MaximizeAccuracy`, `MinimizePower`, `Balanced`.
- **`LambdaPolicy`** — wrap any callable.
- **`StabilityGuard`** — wraps an inner policy and gates *whether* to swap even when the recommendation changes.

### Swaps and safety

- Failed loads **do not** replace the active backend; **`on_fallback`** carries the error string; `last_swap_reason` uses stable **`SWAP_*`** prefixes (see `python/edgeadapt/swap_codes.py`).
- Swap lifecycle and guarantees are documented in [`docs/swap_lifecycle.md`](docs/swap_lifecycle.md).

### Backends

- Subclass **`InferenceBackend`**: `load`, `infer`, `unload`, `is_loaded`, **`capabilities()`** → **`BackendCapability`**.
- **`ONNXBackend`** implements ONNX Runtime loading, timing for last load, and capability introspection.

### Observability

- **`ObservabilityCollector`** records profile ticks, policy evaluations, swap start/complete/fail, and counters (`swaps_attempted`, `swaps_succeeded`, `swaps_failed`, …). Access via **`engine.observability.snapshot()`**.

---

## CLI reference

| Command | Purpose |
|---------|---------|
| `edgeadapt doctor` | Environment and dependency diagnostics (JSON). |
| `edgeadapt validate --config FILE` | Parse family YAML, list variants, check paths exist. |
| `edgeadapt validate --config FILE --explain` | Same, plus one-shot **MaximizeAccuracy** decision + explanation JSON. |
| `edgeadapt decisions --config FILE` | Policy decision trace (JSON) without full validate lines. |
| `edgeadapt benchmark --config FILE [--runs N] [--sample FILE.npy]` | Per-variant benchmark table or **`--json`** for `BenchmarkReport` v2. |
| `edgeadapt profile` | Stream `DeviceState`-like JSON lines for a duration. |
| `edgeadapt inspect-model PATH.onnx` | ONNX I/O metadata via ONNX Runtime. |

Logging: set **`EDGEADAPT_LOG=DEBUG`** for verbose engine logs.

---

## Production-oriented features

Recent milestone work focuses on **trust**, not feature sprawl:

1. **Swap safety** — documented lifecycle; history recorded only on successful commit; tests for failed load, concurrent infer during swap, repeated swaps, stop under pending load.
2. **Observability** — structured events and counters; policy explanations for presets.
3. **Backend contract** — `BackendCapability`; engine can **skip** variants that fail `can_activate_variant` (default: only `backend=="onnx"` until other loaders are wired).
4. **Stress / failure tests** — expanded matrix in `tests/python/test_matrix_failures.py`, `test_swap_matrix.py`, etc.
5. **Benchmark credibility** — p50/p95/p99, warmup/timed counts, versions, optional git commit, input shapes; [`docs/benchmarking.md`](docs/benchmarking.md).

---

## Configuration and extension points

- **Custom loader** — `Engine(..., loader=callable)` returns a loaded `InferenceBackend` (testing or alternate backends).
- **`can_activate_variant`** — `Callable[[ModelVariant], bool]` to gate swaps (e.g. capability or fleet policy).
- **Profiler** — inject `PlatformProfiler` for tests or custom signal sources.
- **Stability** — inject `StabilityGuard` with tuned hysteresis/cooldown/frequency limits.
- **Callbacks** — `on_swap_start`, `on_swap_complete`, `on_fallback` forwarded to the hot swapper.

Contributing and extension conventions: [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Benchmarking

The benchmark runner measures per-infer latency distributions and attaches run metadata. Use the CLI **`--json`** flag for machine-readable output suitable for CI artifacts. Methodology and fields are described in [`docs/benchmarking.md`](docs/benchmarking.md).

---

## Documentation

- **MkDocs** (Material theme): sources under [`docs/`](docs/).
- Build locally: `make docs` (after `pip install mkdocs-material` or project `[dev]` extras).

Key pages:

- [`docs/architecture.md`](docs/architecture.md) — layer diagram and responsibilities.
- [`docs/platform.md`](docs/platform.md) — scope vs full CV/ONNX stacks.
- [`docs/roadmap.md`](docs/roadmap.md) — planned direction.
- [`docs/gap_analysis.md`](docs/gap_analysis.md) — honest status vs goals.
- [`docs/swap_lifecycle.md`](docs/swap_lifecycle.md) — swap states and guarantees.
- [`docs/benchmarking.md`](docs/benchmarking.md) — benchmark methodology.

---

## Development

```bash
maturin develop --extras dev
pytest tests/python -q
ruff check python
cargo fmt && cargo clippy
```

CI (GitHub Actions) runs tests and Rust checks—see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Security

See [`SECURITY.md`](SECURITY.md) for reporting vulnerabilities.

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
