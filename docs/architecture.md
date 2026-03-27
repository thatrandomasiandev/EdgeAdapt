# Architecture

EdgeAdapt is organized in four layers:

1. **Profiler** — samples battery, thermals, memory, CPU, and related signals into `DeviceState`.
2. **Registry** — names model variants, file paths, tiers, and benchmark metadata.
3. **Policy** — maps `(DeviceState, ModelFamily)` to a variant name, optionally wrapped by `StabilityGuard`.
4. **Hot swapper** — loads a new variant in a background thread and swaps the active backend atomically on success.

The `Engine` class wires these pieces together and exposes `infer()`.
