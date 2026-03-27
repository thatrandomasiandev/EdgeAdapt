"""Linux profiler: sysfs/proc reads via Rust."""

from __future__ import annotations

import time
from typing import Any

from edgeadapt.profiler.base import DeviceState, PlatformProfiler


def _get_device_state(root: str | None) -> dict[str, Any]:
    from edgeadapt import _edgeadapt_core  # type: ignore[attr-defined]

    return _edgeadapt_core.get_device_state(root)


class LinuxProfiler(PlatformProfiler):
    """Linux-specific profiler backed by `edgeadapt._edgeadapt_core.get_device_state`."""

    def __init__(self, root: str | None = None) -> None:
        """Create a profiler.

        Args:
            root: Optional filesystem root for tests (fake `/proc` and `/sys` trees).
        """
        self._root = root

    def read(self) -> DeviceState:
        data = _get_device_state(self._root)
        state = DeviceState.from_mapping(data)
        state.timestamp = time.monotonic()
        return state
