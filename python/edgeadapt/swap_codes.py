"""Stable string codes for swap outcomes (logs, metrics, `last_swap_reason` prefixes)."""

# Successful activation of a new variant
SWAP_COMMITTED = "SWAP_COMMITTED"

# Loader raised or returned failure; previous variant remains active
SWAP_LOAD_FAILED = "SWAP_LOAD_FAILED"

# Policy requested a swap; load not yet finished (informational)
SWAP_PENDING = "SWAP_PENDING"

# Policy-driven recommendation text (background loop)
SWAP_POLICY_RECOMMEND = "SWAP_POLICY_RECOMMEND"
