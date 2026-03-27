"""Structured engine events, policy decision records, and in-process counters."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineEvent:
    """A single discrete observation from the control plane."""

    kind: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservabilityCollector:
    """Ring buffer of recent events plus swap counters (thread-safe)."""

    max_events: int = 128
    _events: deque[EngineEvent] = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    swaps_attempted: int = 0
    swaps_succeeded: int = 0
    swaps_failed: int = 0
    profile_ticks: int = 0
    policy_evals: int = 0

    def __post_init__(self) -> None:
        self._events = deque(maxlen=self.max_events)

    def on_profile_tick(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.profile_ticks += 1
            self._events.append(EngineEvent("profile_tick", time.time(), payload))

    def on_policy_eval(
        self,
        *,
        current_variant: str,
        recommended: str,
        explanation: dict[str, Any],
        will_swap: bool,
    ) -> None:
        with self._lock:
            self.policy_evals += 1
            self._events.append(
                EngineEvent(
                    "policy_eval",
                    time.time(),
                    {
                        "current_variant": current_variant,
                        "recommended": recommended,
                        "explanation": explanation,
                        "will_swap": will_swap,
                    },
                )
            )

    def on_swap_start(self, variant: str) -> None:
        with self._lock:
            self.swaps_attempted += 1
            self._events.append(EngineEvent("swap_start", time.time(), {"variant": variant}))

    def on_swap_complete(self, variant: str) -> None:
        with self._lock:
            self.swaps_succeeded += 1
            self._events.append(EngineEvent("swap_complete", time.time(), {"variant": variant}))

    def on_swap_fail(self, variant: str, message: str) -> None:
        with self._lock:
            self.swaps_failed += 1
            short = message if len(message) <= 400 else message[:400] + "..."
            self._events.append(
                EngineEvent("swap_fail", time.time(), {"variant": variant, "message": short})
            )

    def recent_events(self, limit: int = 32) -> list[dict[str, Any]]:
        """Return recent events as JSON-serializable dicts."""
        with self._lock:
            snap = list(self._events)[-limit:]
        out: list[dict[str, Any]] = []
        for ev in snap:
            out.append(
                {
                    "kind": ev.kind,
                    "timestamp": ev.timestamp,
                    "payload": ev.payload,
                }
            )
        return out

    def snapshot(self) -> dict[str, Any]:
        """Counters and last few events for diagnostics."""
        with self._lock:
            counters = {
                "swaps_attempted": self.swaps_attempted,
                "swaps_succeeded": self.swaps_succeeded,
                "swaps_failed": self.swaps_failed,
                "profile_ticks": self.profile_ticks,
                "policy_evals": self.policy_evals,
            }
            recent = [
                {"kind": e.kind, "timestamp": e.timestamp, "payload": e.payload}
                for e in list(self._events)[-16:]
            ]
        return {"counters": counters, "recent_events": recent}
