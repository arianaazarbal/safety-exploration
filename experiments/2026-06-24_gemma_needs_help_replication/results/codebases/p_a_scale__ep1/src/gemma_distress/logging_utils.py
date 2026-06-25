"""Structured logging configured once per process.

Logs go to stderr and, when an output directory is set, to a rotating file so
that multi-week unattended runs leave an inspectable trail.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("gemma_distress")
    if _CONFIGURED:
        return root

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # 50 MB x 10 files keeps weeks of logs without unbounded growth.
        fileh = RotatingFileHandler(
            log_dir / "run.log", maxBytes=50 * 1024 * 1024, backupCount=10
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)

    root.propagate = False
    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"gemma_distress.{name}")
