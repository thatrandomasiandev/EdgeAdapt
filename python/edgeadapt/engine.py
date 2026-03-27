"""Top-level `Engine` API."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from edgeadapt.backends.base import InferenceBackend
from edgeadapt.backends.onnx_backend import ONNXBackend
from edgeadapt.benchmark.runner import BenchmarkReport, apply_report_to_family, benchmark_family
from edgeadapt.logging_config import get_logger
from edgeadapt.observability import ObservabilityCollector
from edgeadapt.policy.base import BasePolicy, LambdaPolicy, SwapHistory
from edgeadapt.policy.stability import StabilityGuard
from edgeadapt.profiler.base import DeviceState, PlatformProfiler, get_profiler
from edgeadapt.registry.family import ModelFamily, ModelVariant
from edgeadapt.swap_codes import SWAP_COMMITTED, SWAP_LOAD_FAILED, SWAP_PENDING, SWAP_POLICY_RECOMMEND
from edgeadapt.swapper.swapper import HotSwapper

_LOG = get_logger("engine")


def _default_can_activate_variant(v: ModelVariant) -> bool:
    """Default gate: ONNX-only until additional loaders are wired in ``Engine``."""
    return v.backend == "onnx"


class Engine:
    """Coordinates profiling, policy, and hot-swap."""

    def __init__(
        self,
        family: ModelFamily,
        policy: BasePolicy | Callable[[DeviceState, ModelFamily], str],
        *,
        poll_interval_sec: float = 5.0,
        on_swap_start: Callable[[str], None] | None = None,
        on_swap_complete: Callable[[str], None] | None = None,
        on_fallback: Callable[[str, str], None] | None = None,
        profiler: PlatformProfiler | None = None,
        stability: StabilityGuard | None = None,
        initial_variant: str | None = None,
        loader: Callable[[str], InferenceBackend] | None = None,
        observability: ObservabilityCollector | None = None,
        can_activate_variant: Callable[[ModelVariant], bool] | None = None,
    ) -> None:
        """Create an engine for ``family`` using ``policy``.

        If ``loader`` is provided, it is used instead of the default ONNX load path
        (useful for tests or custom backends). It must return a loaded ``InferenceBackend``.

        ``can_activate_variant`` gates swaps by variant metadata (default: only ``backend=="onnx"``).
        """
        self._family = family
        if isinstance(policy, BasePolicy):
            self._policy: BasePolicy = policy
        else:
            self._policy = LambdaPolicy(policy)
        self._stability = (
            stability
            if stability is not None
            else StabilityGuard(
                self._policy,
                hysteresis=0.05,
                cooldown_sec=30.0,
            )
        )
        self._poll_interval_sec = poll_interval_sec
        self._profiler = profiler or get_profiler()
        self._swap_history = SwapHistory()
        self._last_swap_reason: str | None = None
        self._pending_swap: tuple[str, str] | None = None
        self._user_on_swap_start = on_swap_start
        self._user_on_swap_complete = on_swap_complete
        self._user_on_fallback = on_fallback
        self._obs = observability or ObservabilityCollector()
        self._can_activate_variant = can_activate_variant or _default_can_activate_variant

        names = list(family.variants.keys())
        if not names:
            raise ValueError("ModelFamily must contain at least one variant")
        start_name = initial_variant or names[0]
        if start_name not in family.variants:
            raise ValueError(f"initial_variant {start_name!r} not in family")

        def default_loader(name: str) -> ONNXBackend:
            path = self._family.variants[name].path
            backend = ONNXBackend()
            backend.load(path)
            return backend

        def make_loader(name: str) -> InferenceBackend:
            return loader(name) if loader is not None else default_loader(name)

        def _on_swap_start(name: str) -> None:
            self._obs.on_swap_start(name)
            self._last_swap_reason = f"{SWAP_PENDING}:{name}"
            if self._user_on_swap_start is not None:
                self._user_on_swap_start(name)

        def _on_swap_complete(name: str) -> None:
            self._obs.on_swap_complete(name)
            if self._pending_swap is not None and self._pending_swap[1] == name:
                frm, to = self._pending_swap
                self._swap_history.record(frm, to)
                self._stability.notify_swap(frm, to)
                self._pending_swap = None
            self._last_swap_reason = f"{SWAP_COMMITTED}:{name}"
            if self._user_on_swap_complete is not None:
                self._user_on_swap_complete(name)

        def _on_fallback(variant: str, msg: str) -> None:
            self._obs.on_swap_fail(variant, msg)
            self._pending_swap = None
            short = msg if len(msg) <= 500 else msg[:500] + "..."
            self._last_swap_reason = f"{SWAP_LOAD_FAILED}:{variant}:{short}"
            if self._user_on_fallback is not None:
                self._user_on_fallback(variant, msg)

        self._swapper = HotSwapper(
            make_loader,
            start_name,
            on_swap_start=_on_swap_start,
            on_swap_complete=_on_swap_complete,
            on_fallback=_on_fallback,
        )

        self._device_state: DeviceState = self._profiler.read()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background profiling + policy evaluation loop."""
        if self._thread and self._thread.is_alive():
            return

        def loop() -> None:
            while not self._stop.wait(self._poll_interval_sec):
                self._device_state = self._profiler.read()
                ds = self._device_state
                self._obs.on_profile_tick(
                    {
                        "available_ram_mb": ds.available_ram_mb,
                        "cpu_utilization_pct": ds.cpu_utilization_pct,
                    }
                )
                recommended, explanation = self._stability.select_with_explanation(
                    self._device_state, self._family
                )
                current = self._swapper.active_variant
                will_swap = self._stability.should_swap(
                    current,
                    recommended,
                    self._swap_history,
                    family=self._family,
                )
                self._obs.on_policy_eval(
                    current_variant=current,
                    recommended=recommended,
                    explanation=dict(explanation),
                    will_swap=will_swap,
                )
                if will_swap:
                    cand = self._family.variants[recommended]
                    if not self._can_activate_variant(cand):
                        self._last_swap_reason = f"SWAP_SKIPPED_CAPABILITY:{recommended}"
                        continue
                    self._last_swap_reason = f"{SWAP_POLICY_RECOMMEND}:{recommended}"
                    _LOG.debug(
                        "swap %s -> %s (state ram_mb=%.1f cpu=%.1f)",
                        current,
                        recommended,
                        self._device_state.available_ram_mb,
                        self._device_state.cpu_utilization_pct,
                    )
                    self._pending_swap = (current, recommended)
                    self._swapper.swap_to(recommended)

        self._thread = threading.Thread(target=loop, name="edgeadapt-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background loop."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def infer(self, input_data: Any) -> Any:
        """Run inference on the currently active variant."""
        backend = self._swapper.get_backend()
        if backend is None:
            raise RuntimeError("no active backend loaded yet")
        return backend.infer(input_data)

    @property
    def active_variant(self) -> str:
        """Currently active variant name."""
        return self._swapper.active_variant

    @property
    def last_swap_reason(self) -> str | None:
        """Human-readable reason for the last swap, if any."""
        return self._last_swap_reason

    @property
    def device_state(self) -> DeviceState:
        """Latest observed device state."""
        return self._device_state

    @property
    def observability(self) -> ObservabilityCollector:
        """Structured events and counters for diagnostics."""
        return self._obs

    def benchmark(self, input_sample: Any | None = None, runs: int = 100) -> BenchmarkReport:
        """Benchmark all variants and return a report; metadata is updated on the family."""
        report = benchmark_family(self._family, input_sample, runs=runs)
        self._family = apply_report_to_family(self._family, report)
        return report

    def __enter__(self) -> Engine:
        """Start the engine on context entry."""
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Stop the engine on context exit."""
        self.stop()
