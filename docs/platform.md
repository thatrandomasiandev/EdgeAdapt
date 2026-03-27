# Platform scope

## What EdgeAdapt is

EdgeAdapt is a **runtime policy and orchestration** library:

- **Profiler** — normalized `DeviceState` (Linux via Rust; psutil fallback elsewhere).
- **Registry** — named `ModelFamily` with tiered variants and benchmark metadata.
- **Policy** — selects variants under constraints; `StabilityGuard` reduces thrashing.
- **Hot swapper** — loads a new variant in a background thread and swaps the active backend safely.
- **Engine** — ties the loop together.

Inference itself is delegated to **pluggable backends** (today: ONNX Runtime).

## What EdgeAdapt is not

- A replacement for **ONNX Runtime**, PyTorch, TensorFlow, or OpenCV’s DNN module.
- A **graph optimizer**, **kernel library**, or **broad operator implementation**.
- A guarantee of **numerical parity** across backends (that is a backend and model concern).

## Why this matters strategically

“OpenCV-level” breadth comes from **decades** of cross-cutting investment. A new project should **choose a wedge** and win a specific workflow first. EdgeAdapt’s wedge is **adaptive edge inference orchestration**, not reimplementing ML frameworks.

Use the [Roadmap](roadmap.md) for how we grow **platform** surfaces (tooling, compatibility, governance) without diluting that focus.
