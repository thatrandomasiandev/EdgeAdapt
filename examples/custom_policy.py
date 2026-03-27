"""Custom policy via `LambdaPolicy` and a raw callable accepted by `Engine`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import edgeadapt as ea
from edgeadapt.policy import LambdaPolicy, MaximizeAccuracy
from edgeadapt.profiler.base import DeviceState
from edgeadapt.registry import ModelFamily

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Demonstrate callable policies."""
    family = ModelFamily.from_yaml(str(ROOT / "family.yaml")).resolve_paths(ROOT)

    def choose(state: DeviceState, fam: ModelFamily) -> str:
        if state.battery_pct is not None and state.battery_pct < 10.0:
            return "tiny"
        return "full"

    eng = ea.Engine(family, LambdaPolicy(choose), initial_variant="tiny")
    x = np.random.randn(1, 4).astype(np.float32)
    print("lambda_policy", eng.infer(x).shape)

    eng2 = ea.Engine(family, choose, initial_variant="tiny")
    print("callable", eng2.infer(x).shape)

    eng3 = ea.Engine(family, MaximizeAccuracy(latency_ceiling_ms=200.0), initial_variant="tiny")
    print("preset", eng3.infer(x).shape)


if __name__ == "__main__":
    main()
