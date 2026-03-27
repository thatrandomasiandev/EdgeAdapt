"""Runtime environment and dependency introspection."""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any

import numpy as np

from edgeadapt.version import __version__


def _extension_loaded() -> bool:
    """True if the native ``edgeadapt._edgeadapt_core`` extension is importable."""
    spec = importlib.util.find_spec("edgeadapt._edgeadapt_core")
    return spec is not None


def environment_report() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the runtime environment.

    Used by ``edgeadapt doctor`` and bug reports.
    """
    ort_version: str | None = None
    try:
        import onnxruntime as ort  # noqa: PLC0415

        ort_version = getattr(ort, "__version__", None)
    except Exception:
        ort_version = None

    return {
        "edgeadapt_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "onnxruntime_version": ort_version,
        "rust_extension_loaded": _extension_loaded(),
    }
