"""Profiler tests."""

from __future__ import annotations

import sys

import pytest

from edgeadapt.profiler.base import DeviceState, FallbackProfiler, get_profiler


def test_fallback_profiler_shape() -> None:
    """Fallback profiler returns a populated DeviceState."""
    fb = FallbackProfiler()
    s = fb.read()
    assert isinstance(s, DeviceState)
    assert s.available_ram_mb > 0


def test_get_profiler_factory() -> None:
    """Factory returns a profiler instance."""
    p = get_profiler()
    s = p.read()
    assert s.timestamp is not None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only Rust path")
def test_linux_profiler_import() -> None:
    """Linux profiler uses the Rust extension when available."""
    from edgeadapt.profiler.linux import LinuxProfiler

    _ = LinuxProfiler()
