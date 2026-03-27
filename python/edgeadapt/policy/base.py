"""Policy base types and lambda wrapper."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable

from edgeadapt.profiler.base import DeviceState
from edgeadapt.registry.family import ModelFamily


class SwapHistory:
    """Ring buffer of recent swap events."""

    def __init__(self, max_events: int = 32) -> None:
        self._max = max_events
        self._events: deque[tuple[float, str, str]] = deque(maxlen=max_events)

    def record(self, from_variant: str, to_variant: str) -> None:
        """Record a swap at the current monotonic time."""
        self._events.append((time.monotonic(), from_variant, to_variant))

    def count_since(self, seconds: float) -> int:
        """Count swaps in the last ``seconds``."""
        cutoff = time.monotonic() - seconds
        return sum(1 for ts, _, _ in self._events if ts >= cutoff)

    def __len__(self) -> int:
        return len(self._events)


class BasePolicy(ABC):
    """Selects a variant name given device state and a model family."""

    @abstractmethod
    def select(self, state: DeviceState, family: ModelFamily) -> str:
        """Return the variant name to use."""

    def select_with_explanation(
        self, state: DeviceState, family: ModelFamily
    ) -> tuple[str, dict[str, object]]:
        """Return ``(variant_name, explanation_dict)`` for observability and debugging."""
        return self.select(state, family), {}

    def should_swap(
        self,
        current: str,
        recommended: str,
        history: SwapHistory,
        *,
        family: ModelFamily | None = None,
    ) -> bool:
        """Default anti-thrashing hook; `family` is provided by `Engine` when available."""
        _ = history, family
        return current != recommended


class LambdaPolicy(BasePolicy):
    """Wraps a ``(state, family) -> variant_name`` callable."""

    def __init__(self, fn: Callable[[DeviceState, ModelFamily], str]) -> None:
        self._fn = fn

    def select(self, state: DeviceState, family: ModelFamily) -> str:
        """Delegate to the wrapped callable."""
        return self._fn(state, family)

    def select_with_explanation(
        self, state: DeviceState, family: ModelFamily
    ) -> tuple[str, dict[str, object]]:
        """Include the callable's ``__name__`` when available."""
        name = getattr(self._fn, "__name__", "lambda")
        return self._fn(state, family), {"policy": "LambdaPolicy", "callable": name}
