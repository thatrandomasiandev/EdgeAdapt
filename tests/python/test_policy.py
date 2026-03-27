"""Policy engine tests."""

from __future__ import annotations

import time

from edgeadapt.policy.base import SwapHistory
from edgeadapt.policy.presets import Balanced, MaximizeAccuracy, MinimizePower
from edgeadapt.policy.stability import StabilityGuard
from edgeadapt.profiler.base import DeviceState
from edgeadapt.registry.family import ModelFamily, VariantMetadata


def _family() -> ModelFamily:
    fam = ModelFamily(name="f")
    fam.add_variant(
        "full",
        "a.onnx",
        "high",
        metadata=VariantMetadata(120, 250, 0.92, "high"),
    )
    fam.add_variant(
        "tiny",
        "b.onnx",
        "low",
        metadata=VariantMetadata(15, 20, 0.74, "low"),
    )
    return fam


def test_maximize_accuracy() -> None:
    """Higher accuracy variant is preferred when RAM allows."""
    fam = _family()
    state = DeviceState(
        battery_pct=50.0,
        is_charging=True,
        thermal_zone_c=40.0,
        available_ram_mb=500.0,
        cpu_utilization_pct=10.0,
        gpu_utilization_pct=None,
        network_quality="good",
        timestamp=time.monotonic(),
    )
    p = MaximizeAccuracy(latency_ceiling_ms=200.0)
    assert p.select(state, fam) == "full"


def test_minimize_power() -> None:
    """Low-power variant is preferred."""
    fam = _family()
    state = DeviceState(None, None, None, 100.0, 5.0, None, None, time.monotonic())
    p = MinimizePower(min_accuracy_floor=0.7)
    assert p.select(state, fam) == "tiny"


def test_balanced() -> None:
    """Balanced selects a feasible variant."""
    fam = _family()
    state = DeviceState(None, None, None, 100.0, 5.0, None, None, time.monotonic())
    p = Balanced(latency_ceiling_ms=200.0)
    name = p.select(state, fam)
    assert name in fam.variants


def test_stability_guard_thrash() -> None:
    """Rapid oscillation should not always swap when frequency cap triggers."""
    fam = _family()
    inner = MaximizeAccuracy(latency_ceiling_ms=500.0)
    guard = StabilityGuard(
        inner, hysteresis=0.01, cooldown_sec=3600.0, max_swaps_per_window=1, swap_window_sec=60.0
    )
    hist = SwapHistory()
    # Force many "swaps" in history
    for _ in range(5):
        hist.record("full", "tiny")
    assert guard.should_swap("full", "tiny", hist, family=fam) is False
