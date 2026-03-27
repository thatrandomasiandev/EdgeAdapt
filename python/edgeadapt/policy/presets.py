"""Built-in policy presets."""

from __future__ import annotations

from edgeadapt.policy.base import BasePolicy
from edgeadapt.profiler.base import DeviceState
from edgeadapt.registry.family import ModelFamily, ModelVariant

_POWER_ORDER = {"low": 0, "medium": 1, "high": 2}


def _fits_resources(
    v: ModelVariant,
    *,
    available_ram_mb: float,
    latency_ceiling_ms: float | None,
) -> bool:
    return v.metadata.memory_footprint_mb <= available_ram_mb and (
        latency_ceiling_ms is None or v.metadata.expected_latency_ms <= latency_ceiling_ms
    )


class MaximizeAccuracy(BasePolicy):
    """Pick the highest-accuracy variant that fits resource ceilings."""

    def __init__(self, *, latency_ceiling_ms: float | None = None) -> None:
        self._latency_ceiling_ms = latency_ceiling_ms

    def select(self, state: DeviceState, family: ModelFamily) -> str:
        """Sort by descending accuracy and pick the first feasible variant."""
        variants = list(family.variants.values())
        variants.sort(key=lambda x: x.metadata.accuracy_score, reverse=True)
        for v in variants:
            if _fits_resources(
                v,
                available_ram_mb=state.available_ram_mb,
                latency_ceiling_ms=self._latency_ceiling_ms,
            ):
                return v.name
        return variants[0].name if variants else next(iter(family.variants.keys()))


class MinimizePower(BasePolicy):
    """Pick the lowest-power variant that meets a minimum accuracy floor."""

    def __init__(self, *, min_accuracy_floor: float = 0.5) -> None:
        if not 0.0 <= min_accuracy_floor <= 1.0:
            raise ValueError("min_accuracy_floor must be in [0, 1]")
        self._min_accuracy_floor = min_accuracy_floor

    def select(self, state: DeviceState, family: ModelFamily) -> str:
        """Sort by ascending power draw and pick the first above the accuracy floor."""
        _ = state
        variants = [
            v
            for v in family.variants.values()
            if v.metadata.accuracy_score >= self._min_accuracy_floor
        ]
        if not variants:
            variants = list(family.variants.values())
        variants.sort(key=lambda x: _POWER_ORDER.get(x.metadata.power_draw_estimate, 99))
        return variants[0].name


class Balanced(BasePolicy):
    """Weighted score across accuracy, latency slack, and power preference."""

    def __init__(
        self,
        *,
        latency_ceiling_ms: float,
        w_acc: float = 0.5,
        w_lat: float = 0.3,
        w_pwr: float = 0.2,
    ) -> None:
        if latency_ceiling_ms <= 0:
            raise ValueError("latency_ceiling_ms must be positive")
        total = w_acc + w_lat + w_pwr
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self._latency_ceiling_ms = latency_ceiling_ms
        self._w_acc = w_acc / total
        self._w_lat = w_lat / total
        self._w_pwr = w_pwr / total

    def select(self, state: DeviceState, family: ModelFamily) -> str:
        """Pick the highest composite score among feasible variants."""
        best_name: str | None = None
        best_score = float("-inf")
        n = max(1, len(family.variants))
        for v in family.variants.values():
            if not _fits_resources(
                v,
                available_ram_mb=state.available_ram_mb,
                latency_ceiling_ms=self._latency_ceiling_ms,
            ):
                continue
            lat_ratio = min(1.0, v.metadata.expected_latency_ms / self._latency_ceiling_ms)
            lat_term = 1.0 - lat_ratio
            pr = _POWER_ORDER.get(v.metadata.power_draw_estimate, 1)
            pwr_term = 1.0 - (pr / max(3, n))
            score = (
                self._w_acc * v.metadata.accuracy_score
                + self._w_lat * lat_term
                + self._w_pwr * pwr_term
            )
            if score > best_score:
                best_score = score
                best_name = v.name
        if best_name is not None:
            return best_name
        return next(iter(family.variants.keys()))
