"""Shared validation constants for registry types."""

# Bump when `family` YAML shape or semantics change incompatibly.
FAMILY_SCHEMA_VERSION = 1

ALLOWED_TIERS = frozenset({"high", "medium", "low"})
ALLOWED_BACKENDS = frozenset({"onnx", "tflite", "coreml"})
ALLOWED_POWER = frozenset({"high", "medium", "low"})
