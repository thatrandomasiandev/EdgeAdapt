"""Abstract inference backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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
