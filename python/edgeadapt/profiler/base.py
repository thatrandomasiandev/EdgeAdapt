"""Profiler abstractions and fallback implementation."""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class DeviceState:
    """Normalized snapshot of device constraints."""

    battery_pct: float | None
    is_charging: bool | None
    thermal_zone_c: float | None
    available_ram_mb: float
    cpu_utilization_pct: float
    gpu_utilization_pct: float | None
    network_quality: str | None
    timestamp: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DeviceState:
        """Build a `DeviceState` from a mapping (e.g. Rust `get_device_state` dict)."""
        return cls(
            battery_pct=_optional_float(data.get("battery_pct")),
            is_charging=_optional_bool(data.get("is_charging")),
            thermal_zone_c=_optional_float(data.get("thermal_zone_c")),
            available_ram_mb=float(data.get("available_ram_mb", 0.0)),
            cpu_utilization_pct=float(data.get("cpu_utilization_pct", 0.0)),
            gpu_utilization_pct=_optional_float(data.get("gpu_utilization_pct")),
            network_quality=_optional_str(data.get("network_quality")),
            timestamp=float(data.get("timestamp", 0.0)),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


class PlatformProfiler(ABC):
    """Abstract base for platform-specific profilers."""

    @abstractmethod
    def read(self) -> DeviceState:
        """Return a fresh `DeviceState`."""


def get_profiler() -> PlatformProfiler:
    """Return the best profiler for the current platform."""
    if sys.platform.startswith("linux"):
        from edgeadapt.profiler.linux import LinuxProfiler

        return LinuxProfiler()
    return FallbackProfiler()


class FallbackProfiler(PlatformProfiler):
    """Development profiler using psutil where sysfs is unavailable."""

    def read(self) -> DeviceState:
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        return DeviceState(
            battery_pct=None,
            is_charging=None,
            thermal_zone_c=None,
            available_ram_mb=float(vm.available) / (1024.0 * 1024.0),
            cpu_utilization_pct=float(cpu),
            gpu_utilization_pct=None,
            network_quality=None,
            timestamp=time.monotonic(),
        )
