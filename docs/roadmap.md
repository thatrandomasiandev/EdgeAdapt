# Roadmap

EdgeAdapt is **not** a full tensor runtime, operator library, or OpenCV-scale vision stack. It is an **adaptive orchestration layer** on top of existing inference engines (initially ONNX Runtime), focused on **edge deployment**: device-aware variant selection, stable hot-swap, and clear operational tooling.

The path to “OpenCV-level completeness” for this *category* of library is a **multi-year platform** build-out. Below is a practical phased plan aligned with a **wedge**: *reliable on-device inference with transparent model downgrades/upgrades under real constraints*.

## 0–6 months (harden the wedge)

- Stabilize **family YAML** schema (`schema_version`), compatibility notes, and migration guidance.
- Expand **tests**: profiling with fake sysfs roots, swapper stress tests, policy regression tables.
- **Diagnostics**: environment report, ONNX IO inspection, structured logging (`EDGEADAPT_LOG`).
- **Docs**: tutorials, troubleshooting, performance tuning guidance for variant metadata.
- **CI**: keep Python 3.10+ and Rust clippy/fmt green on Linux; macOS/Windows smoke where feasible.

## 6–12 months (platform depth)

- Additional **backends** behind `InferenceBackend` (e.g. TFLite, Core ML) as optional extras.
- **Platform profilers**: Android/iOS phase work behind stable interfaces.
- **Benchmark suite**: reproducible scripts per device class, stored reports format.
- **Tooling**: richer `validate` (opset checks, IO shape hints), optional integration with external model checkers.

## 12–24 months (ecosystem and governance)

- **Versioning policy** for YAML and public Python APIs; deprecation windows.
- **Security process** (see `SECURITY.md`), coordinated disclosure, release notes for CVE-related fixes.
- **Bindings**: evaluate thin C ABI or gRPC sidecar for non-Python runtimes if demand exists.
- **Community**: plugin hooks for custom policies/profilers without forking core.

## What we intentionally defer

- Implementing a custom **graph executor**, **large operator coverage**, or **replacing ONNX Runtime**—those are separate products. EdgeAdapt should integrate and compose them, not duplicate them.

For a deeper discussion of scope, see [Platform scope](platform.md).
