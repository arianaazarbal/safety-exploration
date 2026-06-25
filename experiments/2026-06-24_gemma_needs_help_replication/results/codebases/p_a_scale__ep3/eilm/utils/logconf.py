"""Logging setup: console + rotating file handler, shared across scripts."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(log_dir: str | Path, level: str = "INFO", run_name: Optional[str] = None) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("eilm")
    if _CONFIGURED:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{run_name or 'eilm'}.log"
    fh = logging.handlers.RotatingFileHandler(
        Path(log_dir) / fname, maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    _CONFIGURED = True
    return logger
