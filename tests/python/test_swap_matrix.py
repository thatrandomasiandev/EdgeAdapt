"""Swap failure paths, concurrency, and lifecycle (production milestone)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from edgeadapt.backends.onnx_backend import ONNXBackend
from edgeadapt.engine import Engine
from edgeadapt.policy.base import LambdaPolicy
from edgeadapt.profiler.base import DeviceState
from edgeadapt.registry.family import ModelFamily, VariantMetadata
from edgeadapt.swap_codes import SWAP_COMMITTED
from edgeadapt.swapper.swapper import HotSwapper


def test_hot_swapper_failed_load_keeps_previous(dummy_model_paths: dict[str, object]) -> None:
    """Loader error does not change active variant or backend."""
    paths = {k: str(v) for k, v in dummy_model_paths.items()}
    fallback: list[tuple[str, str]] = []

    def loader(name: str) -> ONNXBackend:
        if name == "bad":
            raise RuntimeError("intentional load failure")
        b = ONNXBackend()
        b.load(paths[name])
        return b

    hs = HotSwapper(
        loader,
        "low",
        on_fallback=lambda v, m: fallback.append((v, m)),
    )
    assert hs.active_variant == "low"
    hs.swap_to("bad")
    time.sleep(0.3)
    assert hs.active_variant == "low"
    assert fallback and fallback[0][0] == "bad"


def test_engine_failed_policy_swap_keeps_last_known_good(dummy_model_paths: dict[str, object]) -> None:
    """Engine stays on initial variant when target path is invalid."""
    bad_path = str(Path(dummy_model_paths["low"]).parent / "does_not_exist.onnx")
    fam = ModelFamily(name="t")
    fam.add_variant(
        "low",
        str(dummy_model_paths["low"]),
        "low",
        metadata=VariantMetadata(15, 20, 0.74, "low"),
    )
    fam.add_variant(
        "broken",
        bad_path,
        "high",
        metadata=VariantMetadata(1, 1, 0.99, "high"),
    )

    def always_broken(_s: DeviceState, _f: ModelFamily) -> str:
        return "broken"

    state = DeviceState(None, None, None, 10_000.0, 1.0, None, None, time.monotonic())

    class OneStateProfiler:
        def read(self) -> DeviceState:
            return state

    eng = Engine(
        fam,
        LambdaPolicy(always_broken),
        poll_interval_sec=0.05,
        profiler=OneStateProfiler(),
        stability=None,
        initial_variant="low",
    )
    eng.start()
    time.sleep(0.35)
    assert eng.active_variant == "low"
    assert eng.last_swap_reason is not None
    assert eng.last_swap_reason.startswith("SWAP_LOAD_FAILED")
    eng.stop()


def test_concurrent_infer_during_swap(dummy_model_paths: dict[str, object]) -> None:
    """infer() remains callable while a background swap is in flight."""
    paths = {k: str(v) for k, v in dummy_model_paths.items()}
    swap_started = threading.Event()

    def loader(name: str) -> ONNXBackend:
        if name == "medium":
            swap_started.wait(timeout=2.0)
            time.sleep(0.05)
        # Always load the same ONNX file so input shape stays consistent under load races.
        b = ONNXBackend()
        b.load(paths["low"])
        return b

    hs = HotSwapper(loader, "low")
    x = np.random.randn(1, 4).astype(np.float32)
    errors: list[BaseException] = []

    def infer_loop() -> None:
        for _ in range(50):
            try:
                b = hs.get_backend()
                if b is not None:
                    b.infer(x)
            except BaseException as e:
                errors.append(e)
            time.sleep(0.002)

    t_inf = threading.Thread(target=infer_loop, daemon=True)
    t_inf.start()
    hs.swap_to("medium")
    swap_started.set()
    time.sleep(0.4)
    t_inf.join(timeout=2.0)
    assert not errors


def test_repeated_swaps_settle(dummy_model_paths: dict[str, object]) -> None:
    """Multiple swap_to calls eventually leave a valid backend."""
    paths = {k: str(v) for k, v in dummy_model_paths.items()}

    def loader(name: str) -> ONNXBackend:
        b = ONNXBackend()
        b.load(paths[name])
        return b

    hs = HotSwapper(loader, "low")
    for _ in range(3):
        hs.swap_to("medium")
        time.sleep(0.15)
        hs.swap_to("low")
        time.sleep(0.15)
    assert hs.active_variant in ("low", "medium")
    assert hs.get_backend() is not None


def test_engine_stop_does_not_hang_with_pending_swap(dummy_model_paths: dict[str, object]) -> None:
    """Stopping the engine while a swap is slow should return promptly."""
    paths = {k: str(v) for k, v in dummy_model_paths.items()}
    proceed = threading.Event()

    def slow_loader(name: str) -> ONNXBackend:
        if name == "high":
            proceed.wait(timeout=5.0)
        b = ONNXBackend()
        b.load(paths[name])
        return b

    fam = ModelFamily(name="t")
    fam.add_variant(
        "low",
        str(dummy_model_paths["low"]),
        "low",
        metadata=VariantMetadata(15, 20, 0.74, "low"),
    )
    fam.add_variant(
        "high",
        str(dummy_model_paths["high"]),
        "high",
        metadata=VariantMetadata(120, 250, 0.92, "high"),
    )

    def pick_high(_s: DeviceState, _f: ModelFamily) -> str:
        return "high"

    state = DeviceState(None, None, None, 10_000.0, 1.0, None, None, time.monotonic())

    class OneStateProfiler:
        def read(self) -> DeviceState:
            return state

    eng = Engine(
        fam,
        LambdaPolicy(pick_high),
        poll_interval_sec=0.05,
        profiler=OneStateProfiler(),
        stability=None,
        initial_variant="low",
        loader=slow_loader,
    )
    eng.start()
    time.sleep(0.08)
    eng.stop()
    proceed.set()
    assert eng.active_variant == "low"


def test_successful_swap_records_committed_reason(dummy_model_paths: dict[str, object]) -> None:
    """After a successful swap, last_swap_reason reflects commit."""
    paths = {k: str(v) for k, v in dummy_model_paths.items()}

    def loader(name: str) -> ONNXBackend:
        b = ONNXBackend()
        b.load(paths[name])
        return b

    fam = ModelFamily(name="t")
    fam.add_variant(
        "low",
        str(dummy_model_paths["low"]),
        "low",
        metadata=VariantMetadata(15, 20, 0.74, "low"),
    )
    fam.add_variant(
        "high",
        str(dummy_model_paths["high"]),
        "high",
        metadata=VariantMetadata(120, 250, 0.92, "high"),
    )

    def pick_high(_s: DeviceState, _f: ModelFamily) -> str:
        return "high"

    state = DeviceState(None, None, None, 10_000.0, 1.0, None, None, time.monotonic())

    class OneStateProfiler:
        def read(self) -> DeviceState:
            return state

    eng = Engine(
        fam,
        LambdaPolicy(pick_high),
        poll_interval_sec=0.05,
        profiler=OneStateProfiler(),
        stability=None,
        initial_variant="low",
    )
    eng.start()
    time.sleep(0.5)
    eng.stop()
    assert eng.active_variant == "high"
    assert eng.last_swap_reason is not None
    assert SWAP_COMMITTED in eng.last_swap_reason
