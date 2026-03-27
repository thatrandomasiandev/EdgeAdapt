"""Centralized logging configuration for EdgeAdapt."""

from __future__ import annotations

import logging
import os
import sys

_LOG_ENV_VAR = "EDGEADAPT_LOG"
_CONFIGURED = False


def _level_from_name(name: str) -> int:
    """Map a level name to a ``logging`` level constant."""
    level = getattr(logging, name.upper(), None)
    if isinstance(level, int):
        return level
    return logging.WARNING


def configure_logging(
    *,
    level: int | None = None,
    stream: object | None = None,
) -> None:
    """Configure the root EdgeAdapt logger (idempotent).

    If ``level`` is omitted, reads ``EDGEADAPT_LOG`` (e.g. ``DEBUG``, ``INFO``).
    """
    global _CONFIGURED
    if level is None:
        level = _level_from_name(os.environ.get(_LOG_ENV_VAR, "WARNING"))
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )
    root = logging.getLogger("edgeadapt")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under ``edgeadapt``; configures logging once on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"edgeadapt.{name}")
