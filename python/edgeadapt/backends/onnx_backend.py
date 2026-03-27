"""ONNX Runtime inference backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnxruntime as ort

from edgeadapt.backends.base import InferenceBackend


class ONNXBackend(InferenceBackend):
    """Wraps `onnxruntime.InferenceSession`."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._path: str | None = None

    def load(self, path: str) -> None:
        """Load an ONNX model from ``path``."""
        try:
            self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        except FileNotFoundError:
            raise
        except Exception as e:
            raise OSError(f"failed to load ONNX model at {path!r}: {type(e).__name__}: {e}") from e
        self._path = path

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
