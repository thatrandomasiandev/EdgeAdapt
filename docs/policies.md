# Policies

Built-in presets:

- `MaximizeAccuracy` — prefers higher `accuracy_score` subject to RAM and latency ceilings.
- `MinimizePower` — prefers lower `power_draw_estimate` subject to a minimum accuracy floor.
- `Balanced` — weighted score across accuracy, latency slack, and power preference.

Custom policies can subclass `BasePolicy` or be passed as callables wrapped by `LambdaPolicy`.

Anti-thrashing is implemented by `StabilityGuard` (hysteresis on accuracy deltas, cooldown, and swap-frequency limits).
