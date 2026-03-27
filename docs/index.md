# EdgeAdapt

EdgeAdapt selects among registered ML model variants at runtime based on device constraints, and hot-swaps the active model without blocking inference.

## Features

- Normalized `DeviceState` profiling (Linux via Rust; psutil fallback elsewhere)
- YAML-driven `ModelFamily` registry
- Pluggable policies with anti-thrashing `StabilityGuard`
- ONNX Runtime backend with a Rust-backed hot swapper

See [Quickstart](quickstart.md) to get running in minutes. For how EdgeAdapt relates to larger “platform” expectations (and what is explicitly out of scope), read [Platform scope](platform.md) and the [Roadmap](roadmap.md).
