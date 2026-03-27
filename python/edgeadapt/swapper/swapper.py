"""Python wrapper around the Rust `HotSwapper`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from edgeadapt.backends.base import InferenceBackend


def _rust_hot_swapper() -> Any:
    from edgeadapt import _edgeadapt_core  # type: ignore[attr-defined]

    return _edgeadapt_core.HotSwapper


class HotSwapper:
    """Coordinates background loads and atomic backend swaps."""

    def __init__(
        self,
        loader: Callable[[str], InferenceBackend],
        initial_variant: str,
        *,
        on_swap_start: Callable[[str], None] | None = None,
        on_swap_complete: Callable[[str], None] | None = None,
        on_fallback: Callable[[str, str], None] | None = None,
    ) -> None:
        cls = _rust_hot_swapper()
        self._inner = cls(
            loader,
            initial_variant,
            on_swap_start,
            on_swap_complete,
            on_fallback,
        )
        self._inner.load_initial(initial_variant)

    @property
    def active_variant(self) -> str:
        """Name of the currently active variant."""
        return self._inner.active_variant

    def get_backend(self) -> InferenceBackend | None:
        """Return the active backend object, if any."""
        b = self._inner.get_backend()
        return b

    def swap_to(self, variant: str) -> None:
        """Begin loading ``variant`` in the background and swap when ready."""
        self._inner.swap_to(variant)
