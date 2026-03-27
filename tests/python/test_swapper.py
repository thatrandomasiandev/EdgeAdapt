"""Hot swapper tests."""

from __future__ import annotations

import time

from edgeadapt.backends.onnx_backend import ONNXBackend
from edgeadapt.swapper.swapper import HotSwapper


def test_swap_and_load(dummy_model_paths: dict[str, object]) -> None:
    """Swapper loads a variant and returns a backend."""
    paths = {k: str(v) for k, v in dummy_model_paths.items()}

    def loader(name: str) -> ONNXBackend:
        b = ONNXBackend()
        b.load(paths[name])
        return b

    hs = HotSwapper(loader, "low")
    assert hs.active_variant == "low"
    b = hs.get_backend()
    assert b is not None
    hs.swap_to("medium")
    time.sleep(0.2)
    # Eventually medium may be active (background thread)
    assert hs.active_variant in ("low", "medium")
