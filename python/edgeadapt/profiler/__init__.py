"""Device profiling: normalized `DeviceState` and platform adapters."""

from edgeadapt.profiler.base import DeviceState, FallbackProfiler, PlatformProfiler, get_profiler

__all__ = [
    "DeviceState",
    "FallbackProfiler",
    "PlatformProfiler",
    "get_profiler",
]
