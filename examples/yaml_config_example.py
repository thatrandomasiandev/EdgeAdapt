"""Load a model family from YAML and run one inference call."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import edgeadapt as ea
from edgeadapt.policy import Balanced
from edgeadapt.registry import ModelFamily

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Benchmark-oriented policy with a small random input."""
    family = ModelFamily.from_yaml(str(ROOT / "family.yaml")).resolve_paths(ROOT)
    eng = ea.Engine(family, Balanced(latency_ceiling_ms=200.0), initial_variant="tiny")
    x = np.random.randn(1, 4).astype(np.float32)
    print(np.asarray(eng.infer(x)).shape)


if __name__ == "__main__":
    main()
