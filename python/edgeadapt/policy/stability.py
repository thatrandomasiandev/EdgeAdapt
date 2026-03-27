"""Anti-thrashing stability guard wrapping any `BasePolicy`."""

from __future__ import annotations

import time

from edgeadapt.policy.base import BasePolicy, SwapHistory
from edgeadapt.profiler.base import DeviceState
from edgeadapt.registry.family import ModelFamily


class StabilityGuard(BasePolicy):
    """Applies hysteresis, cooldown, and swap-frequency limits."""

    def __init__(
        self,
        inner: BasePolicy,
        *,
        hysteresis: float = 0.05,
        cooldown_sec: float = 30.0,
        max_swaps_per_window: int = 4,
        swap_window_sec: float = 60.0,
    ) -> None:
        self._inner = inner
        self._hysteresis = hysteresis
        self._cooldown_sec = cooldown_sec
        self._max_swaps = max_swaps_per_window
        self._window_sec = swap_window_sec
        self._last_swap_at: float | None = None
        self._locked_variant: str | None = None

    def select(self, state: DeviceState, family: ModelFamily) -> str:
        """Return the recommended variant from the inner policy."""
        return self._inner.select(state, family)

    def should_swap(
        self,
        current: str,
        recommended: str,
        history: SwapHistory,
        *,
        family: ModelFamily | None = None,
    ) -> bool:
        """Apply hysteresis, cooldown, and frequency lock."""
        if current == recommended:
            return False

        if self._locked_variant is not None:
            if recommended != self._locked_variant:
                return False
            self._locked_variant = None

        if history.count_since(self._window_sec) >= self._max_swaps:
            self._locked_variant = current
            return False

        if (
            self._last_swap_at is not None
            and time.monotonic() - self._last_swap_at < self._cooldown_sec
        ):
            return False

        return family is None or self._hysteresis_passes(current, recommended, family)

    def notify_swap(self, from_variant: str, to_variant: str) -> None:
        """Record that a swap occurred (used by `Engine`)."""
        self._last_swap_at = time.monotonic()
        _ = from_variant, to_variant

    def _hysteresis_passes(self, current: str, recommended: str, family: ModelFamily) -> bool:
        """Require a minimum accuracy delta before swapping."""
        if current not in family.variants or recommended not in family.variants:
            return True
        cur = family.variants[current].metadata.accuracy_score
        rec = family.variants[recommended].metadata.accuracy_score
        return abs(rec - cur) >= self._hysteresis
