"""Benchmark all variants and print updated metadata latencies."""

from __future__ import annotations

from pathlib import Path

import edgeadapt as ea
from edgeadapt.policy import MaximizeAccuracy
from edgeadapt.registry import ModelFamily

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Run `Engine.benchmark()` with automatically shaped inputs per variant."""
    family = ModelFamily.from_yaml(str(ROOT / "family.yaml")).resolve_paths(ROOT)
    eng = ea.Engine(family, MaximizeAccuracy(latency_ceiling_ms=500.0), initial_variant="tiny")
    rep = eng.benchmark(None, runs=10)
    for k, row in rep.rows.items():
        print(k, row)


if __name__ == "__main__":
    main()
