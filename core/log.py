"""
core/log.py
===========
Centralised logging configuration for the Apple GUI pipeline.

Usage in any script or module:
    from core.log import get_logger
    logger = get_logger(__name__)

    logger.info("Processing session G1")
    logger.warning("Apple #3 excluded: cx_range=850px < 1000px")
    logger.error("Could not open video: G10.MP4")
    logger.debug("Frame 1452: d_area=168.3px, quality=0.94")

The root logger is configured once (idempotent).  Every subsequent call to
get_logger() returns the module-specific child logger without re-configuring
handlers.
"""

import logging
import sys
from pathlib import Path

# ── Format ─────────────────────────────────────────────────────────────────────
_FMT     = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"

_configured = False


def configure_root(level: int = logging.INFO,
                   log_file: str | None = None) -> None:
    """
    Configure the root logger.  Call this once at the top of each runnable
    script.  Safe to call multiple times — subsequent calls are no-ops.

    Parameters
    ----------
    level    : logging level (default INFO).
    log_file : optional path to write a log file in addition to stdout.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(ch)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        root.addHandler(fh)

    _configured = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger.  Configures the root logger with INFO level and
    stdout handler on first call if it has not already been configured.

    Parameters
    ----------
    name  : typically __name__ of the calling module.
    level : override level for this specific logger (default inherits root).
    """
    configure_root()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
