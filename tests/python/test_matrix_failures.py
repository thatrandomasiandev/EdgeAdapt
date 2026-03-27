"""Failure-mode and adversarial cases (stress matrix)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from edgeadapt.engine import Engine
from edgeadapt.policy.base import LambdaPolicy
from edgeadapt.policy.presets import MaximizeAccuracy
from edgeadapt.profiler.base import DeviceState
from edgeadapt.registry.family import ModelFamily, VariantMetadata
from edgeadapt.registry.yaml_loader import FamilyYamlError
from edgeadapt.swapper.swapper import HotSwapper


class _FakeProfiler:
    def __init__(self, states: list[DeviceState]) -> None:
        self._states = states
        self._i = 0

    def read(self) -> DeviceState:
        s = self._states[min(self._i, len(self._states) - 1)]
        self._i += 1
        return s


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    """Invalid YAML structure is rejected."""
    p = tmp_path / "bad.yaml"
    p.write_text("{ not: yaml", encoding="utf-8")
    with pytest.raises(Exception):
        ModelFamily.from_yaml(str(p))


def test_yaml_empty_variants_raises(tmp_path: Path) -> None:
    """Empty variants mapping raises."""
    p = tmp_path / "empty.yaml"
    p.write_text(
        """
schema_version: 1
family: x
variants: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(FamilyYamlError):
        ModelFamily.from_yaml(str(p))


def test_double_swap_hot_swapper(dummy_model_paths: dict[str, object]) -> None:
    """Issuing swap_to twice before the first completes still leaves a valid backend."""
    paths = {k: str(v) for k, v in dummy_model_paths.items()}

    from edgeadapt.backends.onnx_backend import ONNXBackend

    def loader(name: str) -> ONNXBackend:
        b = ONNXBackend()
        b.load(paths[name])
        return b

    hs = HotSwapper(loader, "low")
    hs.swap_to("medium")
    hs.swap_to("high")
    time.sleep(0.5)
    b = hs.get_backend()
    assert b is not None
    assert hs.active_variant in ("low", "medium", "high")


def test_engine_context_manager_stops(dummy_model_paths: dict[str, object]) -> None:
    """Context manager exits cleanly."""
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
    prof = _FakeProfiler(
        [
            DeviceState(None, None, None, 10_000.0, 1.0, None, None, time.monotonic()),
        ]
    )
    with Engine(fam, MaximizeAccuracy(latency_ceiling_ms=200.0), profiler=prof, initial_variant="low") as eng:
        x = np.random.randn(1, 4).astype(np.float32)
        _ = eng.infer(x)
    # stopped after exit
    assert eng._thread is None or not eng._thread.is_alive()


def test_engine_empty_family_raises() -> None:
    """Engine requires at least one variant."""
    fam = ModelFamily(name="empty")
    with pytest.raises(ValueError, match="at least one variant"):
        Engine(fam, LambdaPolicy(lambda s, f: "x"))


def test_observability_counters_increment(dummy_model_paths: dict[str, object]) -> None:
    """Observability records policy and profile activity."""
    fam = ModelFamily(name="t")
    fam.add_variant(
        "low",
        str(dummy_model_paths["low"]),
        "low",
        metadata=VariantMetadata(15, 20, 0.74, "low"),
    )
    state = DeviceState(None, None, None, 10.0, 1.0, None, None, time.monotonic())

    class OneStateProfiler:
        def read(self) -> DeviceState:
            return state

    eng = Engine(
        fam,
        LambdaPolicy(lambda _s, _f: "low"),
        poll_interval_sec=0.05,
        profiler=OneStateProfiler(),
        stability=None,
        initial_variant="low",
    )
    eng.start()
    time.sleep(0.15)
    snap = eng.observability.snapshot()
    assert snap["counters"]["profile_ticks"] >= 1
    assert snap["counters"]["policy_evals"] >= 1
    eng.stop()
