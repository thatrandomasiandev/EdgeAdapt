"""Abstract inference backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendCapability:
    """Describes what a loaded backend can do (used for selection before/during swap)."""

    backend_id: str
    """Logical backend identifier (e.g. ``onnxruntime``)."""

    dynamic_shapes: bool
    """Whether any input dimension is symbolic or unknown at load time."""

    max_batch_size: int
    """Upper bound on batch dim if known; ``0`` if unknown / unconstrained."""

    supports_concurrent_infer: bool
    """Whether concurrent ``infer`` calls are safe (best-effort; ONNX is generally yes)."""

    quantization: str
    """``none``, ``fp16``, ``int8``, or ``unknown``."""

    input_names: tuple[str, ...]
    """Declared input tensor names."""


class InferenceBackend(ABC):
    """Loads a model and runs inference."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk."""

    @abstractmethod
    def infer(self, input_data: Any) -> Any:
        """Run a single forward pass."""

    @abstractmethod
    def unload(self) -> None:
        """Release loaded resources."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether a model is currently loaded."""

    @abstractmethod
    def capabilities(self) -> BackendCapability:
        """Return capability descriptor for the currently loaded model (after ``load``)."""
