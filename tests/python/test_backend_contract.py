"""Backend capability schema and activation gating."""

from __future__ import annotations

import time

import numpy as np

from edgeadapt.backends.base import BackendCapability, InferenceBackend
from edgeadapt.engine import Engine
from edgeadapt.policy.base import LambdaPolicy
from edgeadapt.profiler.base import DeviceState
from edgeadapt.registry.family import ModelFamily, VariantMetadata


class _MockBackend(InferenceBackend):
    """Minimal backend for contract tests."""

    def __init__(self, cap: BackendCapability) -> None:
        self._cap = cap
        self._loaded = False

    def load(self, path: str) -> None:
        self._loaded = True

    def infer(self, input_data: object) -> object:
        return np.asarray([0.0], dtype=np.float32)

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def capabilities(self) -> BackendCapability:
        return self._cap


def test_engine_skips_non_onnx_variant(dummy_model_paths: dict[str, object]) -> None:
    """Default loader is ONNX-only; variants with other ``backend`` tags are skipped."""
    fam = ModelFamily(name="t")
    fam.add_variant(
        "low",
        str(dummy_model_paths["low"]),
        "low",
        backend="onnx",
        metadata=VariantMetadata(15, 20, 0.74, "low"),
    )
    fam.add_variant(
        "high",
        "/tmp/placeholder.tflite",
        "high",
        backend="tflite",
        metadata=VariantMetadata(15, 20, 0.99, "high"),
    )

    def always_high(_s: DeviceState, _f: ModelFamily) -> str:
        return "high"

    state = DeviceState(None, None, None, 10_000.0, 1.0, None, None, time.monotonic())

    class OneStateProfiler:
        def read(self) -> DeviceState:
            return state

    eng = Engine(
        fam,
        LambdaPolicy(always_high),
        poll_interval_sec=0.05,
        profiler=OneStateProfiler(),
        stability=None,
        initial_variant="low",
    )
    eng.start()
    time.sleep(0.2)
    # Must not swap to tflite placeholder
    assert eng.active_variant == "low"
    assert eng.last_swap_reason is not None
    assert "SWAP_SKIPPED_CAPABILITY" in eng.last_swap_reason
    eng.stop()


def test_mock_backend_capabilities() -> None:
    """Mock backend exposes a stable capability descriptor."""
    cap = BackendCapability(
        backend_id="mock",
        dynamic_shapes=False,
        max_batch_size=1,
        supports_concurrent_infer=True,
        quantization="none",
        input_names=("foo",),
    )
    b = _MockBackend(cap)
    b.load("dummy")
    assert b.capabilities().backend_id == "mock"
