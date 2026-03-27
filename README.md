# EdgeAdapt

[![CI](https://github.com/thatrandomasiandev/EdgeAdapt/actions/workflows/ci.yml/badge.svg)](https://github.com/thatrandomasiandev/EdgeAdapt/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**EdgeAdapt** (working name *openedlib*) is an open-source Python + Rust library that selects among registered ML model variants at runtime based on device constraints (battery, thermals, memory, latency budgets) and hot-swaps the active model without blocking inference.

- **Docs:** see [`docs/`](docs/) (MkDocs Material) or run `make docs` after `pip install mkdocs-material`.
- **Architecture:** four layers (profiler, registry, policy, swapper) wired by `edgeadapt.Engine` — overview in [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --extras dev
python -c "import edgeadapt; print(edgeadapt.__version__)"
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for workflow, profilers/backends/policies extension points, and review expectations.

## Platform scope and roadmap

EdgeAdapt is an **adaptive orchestration** layer (policies, variants, hot-swap) on top of inference engines such as ONNX Runtime—not a replacement for a full ONNX or vision stack. See [`docs/platform.md`](docs/platform.md) and [`docs/roadmap.md`](docs/roadmap.md) for how we grow breadth without diluting the wedge.

## Diagnostics

- `edgeadapt doctor` — environment JSON (Python, ONNX Runtime, native extension).
- `edgeadapt inspect-model path/to/model.onnx` — IO names and shapes via ONNX Runtime.
- `EDGEADAPT_LOG=DEBUG` — optional debug logs from the engine loop.

## Security

See [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
# EdgeAdapt
# EdgeAdapt
