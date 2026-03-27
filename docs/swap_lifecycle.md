# Swap lifecycle

EdgeAdapt treats variant activation as a **load-then-commit** operation coordinated by the Rust `HotSwapper` and the Python `Engine`.

## States (conceptual)

| State | Meaning |
|-------|---------|
| `idle` | No swap in progress; active backend serves `infer()`. |
| `loading` | A candidate variant is being loaded in a background thread (`swap_to`). |
| `active` | Last load succeeded; `active_variant` and backend pointer match. |
| `failed` | Last load failed; **previous** backend and variant remain active (last-known-good). |

There is no separate `pending_commit` in the current implementation: commit is **atomic** once the Python loader returns a backend object successfully.

## Transitions

1. **Policy loop** recommends variant `B` while `A` is active.
2. `Engine` sets internal pending `(A → B)` and calls `swap_to(B)`.
3. Background thread runs `loader(B)`:
   - **Success:** Rust stores the new backend, updates `active_variant` to `B`, invokes `on_swap_complete(B)`. The `Engine` records swap history and stability notification **only here**.
   - **Failure:** Rust does **not** replace the active backend or variant name; invokes `on_fallback(B, error)`. Pending swap is cleared without recording a successful swap.

## Guarantees

- A failed load **never** leaves the engine without a backend if one was successfully loaded before.
- `infer()` always uses whatever backend is currently installed under the read lock; during load, the **previous** backend continues to serve until the swap commits.

## Timeouts and cancellation

Full swap timeouts and explicit cancellation are **not** yet enforced in the Rust thread (future work). Loads should be fast; pathological hangs should be addressed by fixing loader/backend behavior.

See also [Gap analysis](gap_analysis.md) for roadmap items.
