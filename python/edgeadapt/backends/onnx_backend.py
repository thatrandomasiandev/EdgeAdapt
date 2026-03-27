"""ONNX Runtime inference backend."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import onnxruntime as ort

from edgeadapt.backends.base import BackendCapability, InferenceBackend


class ONNXBackend(InferenceBackend):
    """Wraps `onnxruntime.InferenceSession`."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._path: str | None = None
        self._load_time_sec: float | None = None

    def load(self, path: str) -> None:
        """Load an ONNX model from ``path``."""
        t0 = time.perf_counter()
        try:
            self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        except FileNotFoundError:
            raise
        except Exception as e:
            raise OSError(f"failed to load ONNX model at {path!r}: {type(e).__name__}: {e}") from e
        self._path = path
        self._load_time_sec = time.perf_counter() - t0

    @property
    def last_load_time_sec(self) -> float | None:
        """Wall time spent in the last successful ``load`` call."""
        return self._load_time_sec

    def infer(self, input_data: Any) -> Any:
        """Run inference; ``input_data`` must match the model's first input."""
        if self._session is None:
            raise RuntimeError("backend is not loaded")
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("model has no inputs")
        name = inputs[0].name
        arr = np.asarray(input_data)
        outputs = self._session.run(None, {name: arr})
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def unload(self) -> None:
        """Release the session."""
        self._session = None
        self._path = None

    def is_loaded(self) -> bool:
        """Return whether a session is active."""
        return self._session is not None

    def capabilities(self) -> BackendCapability:
        """Describe IO and runtime traits of the loaded session."""
        if self._session is None:
            raise RuntimeError("backend is not loaded")
        inputs = self._session.get_inputs()
        names = tuple(i.name for i in inputs)
        dynamic = False
        batch_hint = 0
        if inputs:
            shape = inputs[0].shape
            for dim in shape:
                if dim is None or isinstance(dim, str):
                    dynamic = True
                elif isinstance(dim, int) and dim > 0 and batch_hint == 0:
                    batch_hint = dim
        return BackendCapability(
            backend_id="onnxruntime",
            dynamic_shapes=dynamic,
            max_batch_size=batch_hint,
            supports_concurrent_infer=True,
            quantization="unknown",
            input_names=names,
        )
