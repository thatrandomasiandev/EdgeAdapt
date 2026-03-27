"""Minimal example: load a family and run inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import edgeadapt as ea
from edgeadapt.policy import MaximizeAccuracy
from edgeadapt.registry import ModelFamily

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Run a short inference pass on the lowest tier for a predictable tensor shape."""
    family = ModelFamily.from_yaml(str(ROOT / "family.yaml")).resolve_paths(ROOT)
    eng = ea.Engine(family, MaximizeAccuracy(latency_ceiling_ms=500.0), initial_variant="tiny")
    x = np.random.randn(1, 4).astype(np.float32)
    y = eng.infer(x)
    print("ok", np.asarray(y).shape)


if __name__ == "__main__":
    main()
