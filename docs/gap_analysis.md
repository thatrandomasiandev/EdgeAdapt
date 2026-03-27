# Strategic gap analysis (repo snapshot)

This page compares a **long-term “control plane” maturity** checklist against the **current** EdgeAdapt repository. It is a living reference for maintainers; see [Roadmap](roadmap.md) for phased priorities.

**Summary:** The repo has a real **adaptive inference orchestration** product shape (registry, policy, profiler, backends abstraction, swapper, engine, CLI, docs, CI). Most items below are **partially done or not started**—not a claim that everything is shipped.

---

## 1) Category framing

| Item | Status |
|------|--------|
| Frame as **adaptive inference orchestration**, not a tensor runtime | **Done in docs** — [Platform scope](platform.md), [Roadmap](roadmap.md) |
| Avoid “OpenCV for inference” scope creep (no custom op/tensor runtime) | **Documented**; inference delegated to ONNX Runtime today |

---

## 2) Gap areas

### A. Backend breadth

| Item | Status |
|------|--------|
| Pluggable `InferenceBackend` ABC | **Done** |
| Second backend + tests proving the abstraction | **Not done** — ONNX only today |
| Capability reporting / negotiation | **Not done** |

### B. Model lifecycle

| Item | Status |
|------|--------|
| Family/variant registration + YAML | **Done** |
| `schema_version` + validation | **Done** |
| Formal migration / deprecation policy | **Partial** — process notes in README / SECURITY / roadmap; no formal semver API doc |
| Model integrity checks (e.g. checksums) | **Not done** — `validate` checks paths exist, not file hashes |
| Fallback on failed swap | **Partial** — `on_fallback` in swapper; prior variant remains if load fails |
| Warmup / explicit preload / rollback API | **Partial** — background `swap_to`; no dedicated preload queue or explicit rollback API |

### C. Policy reliability

| Item | Status |
|------|--------|
| Presets + `StabilityGuard` + `LambdaPolicy` | **Done** |
| Explicit objectives, cost signals, trace replay, explore/exploit | **Not done** — policies remain heuristic |

### D. Observability

| Item | Status |
|------|--------|
| Logging (`EDGEADAPT_LOG`, engine debug around swaps) | **Partial** |
| Traces, metrics export, swap counters, latency history | **Not done** |
| Rich “why this variant” explainability | **Partial** — `last_swap_reason` string only |

### E. Compatibility and hardening

| Item | Status |
|------|--------|
| Unit/integration tests + CI | **Done** |
| Linux profiler tests off Linux | **Partial** — skipped where sysfs unavailable |
| Stress: concurrent infer + swap, repeated swaps, shutdown | **Partial** |
| Corrupt YAML / bad metadata | **Partial** — loader validation; not exhaustive |

---

## 3) Phased priorities (from strategic note)

| Phase | Theme | Largely done? |
|-------|--------|----------------|
| **1** — Bulletproof core (rollback, concurrency, preload, metrics, validation, swap-failure E2E) | Foundations exist; **not** a full hardening pass | **No** |
| **2** — Extensibility (second backend, capabilities, policy ↔ capabilities) | | **No** |
| **3** — Production UX (tracing hooks, config layering, env overrides, packaging) | Logging only; no OpenTelemetry-style tracing | **No** |
| **4** — Ecosystem (examples, benchmark suites, compatibility guarantees, plugins) | Examples + docs + CONTRIBUTING | **Partial** |

---

## 4) Bottom line

- **True today:** Control-plane architecture and a coherent wedge ([Platform scope](platform.md)).
- **Still roadmap:** Deep observability, second backend, policy science, lifecycle hardening, and formal compatibility guarantees—as reflected in [Roadmap](roadmap.md).
