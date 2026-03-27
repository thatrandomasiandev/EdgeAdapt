"""Engine integration tests."""

from __future__ import annotations

import time

import numpy as np

from edgeadapt.engine import Engine
from edgeadapt.policy.presets import MaximizeAccuracy
from edgeadapt.profiler.base import DeviceState
from edgeadapt.registry.family import ModelFamily, VariantMetadata


class FakeProfiler:
    """Deterministic profiler for tests."""

    def __init__(self, states: list[DeviceState]) -> None:
        self._states = states
        self._i = 0

    def read(self) -> DeviceState:
        s = self._states[min(self._i, len(self._states) - 1)]
        self._i += 1
        return s


def test_engine_infer(dummy_model_paths: dict[str, object]) -> None:
    """Engine runs inference with a dummy family."""
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

    prof = FakeProfiler(
        [
            DeviceState(None, None, None, 10.0, 1.0, None, None, time.monotonic()),
        ]
    )
    eng = Engine(
        fam,
        MaximizeAccuracy(latency_ceiling_ms=200.0),
        poll_interval_sec=0.05,
        profiler=prof,
        initial_variant="low",
    )
    x = np.random.randn(1, 4).astype(np.float32)
    y = eng.infer(x)
    assert np.asarray(y).shape[0] == 1
