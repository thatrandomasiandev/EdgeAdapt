# Contributing to EdgeAdapt

Thank you for your interest in contributing. This project uses a mixed Python + Rust codebase built with Maturin and PyO3.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --extras dev
make lint
make test
make test-rust
```

## Adding a new platform profiler

1. Implement a Python class under `python/edgeadapt/profiler/` that subclasses `PlatformProfiler`.
2. Optionally add Rust helpers under `rust/edgeadapt_core/src/profiler/` and expose them via PyO3.
3. Register the platform in `get_profiler()` with clear feature detection.
4. Add unit tests with mocked sysfs/proc trees where possible.

## Adding a new inference backend

1. Subclass `InferenceBackend` in `python/edgeadapt/backends/`.
2. Document expected input/output conventions in the class docstring.
3. Add tests that cover load, infer, unload, and error paths.

## Adding a policy preset

1. Subclass `BasePolicy` in `python/edgeadapt/policy/presets.py` (or a new module).
2. Add focused tests under `tests/python/test_policy.py`.
3. Document behavior in `docs/policies.md`.

## Pull requests

- Keep changes focused and tested.
- Run `ruff format`, `ruff check`, `cargo fmt`, and `cargo clippy` before submitting.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).

## Security

See [`SECURITY.md`](SECURITY.md) for reporting vulnerabilities and scope.

## Logging

Set `EDGEADAPT_LOG=DEBUG` (or `INFO`, `WARNING`) to increase verbosity for `edgeadapt.*` loggers.

