# API reference

## `edgeadapt.Engine`

- `start()` / `stop()` — background profiling loop
- `infer(x)` — forward pass through the active variant
- `benchmark(sample, runs=100)` — benchmark variants and refresh metadata
- `active_variant`, `last_swap_reason`, `device_state` — introspection

## `edgeadapt.registry.ModelFamily`

- `from_yaml(path)` — load a family definition
- `add_variant(...)` — register variants programmatically

## CLI

- `edgeadapt benchmark --config family.yaml [--sample input.npy]`
- `edgeadapt profile --interval 1 --duration 30`
- `edgeadapt validate --config family.yaml`
- `edgeadapt doctor` — JSON environment report (versions, native extension)
- `edgeadapt inspect-model model.onnx` — ONNX Runtime input/output metadata

## Logging

Set `EDGEADAPT_LOG` to `DEBUG`, `INFO`, or `WARNING` to tune verbosity for `edgeadapt.*` loggers.
