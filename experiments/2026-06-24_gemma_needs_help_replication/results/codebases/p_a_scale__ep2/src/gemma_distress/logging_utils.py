"""Structured logging configured once, safe for multi-week unattended runs.

Logs go to stderr *and* a rotating file under the run directory so that a crash weeks
into a run still leaves a forensic trail. Import `get_logger` everywhere.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

_CONFIGURED = False


def configure_logging(run_dir: Path | str | None = None, level: str | None = None) -> None:
    """Idempotently configure root logging. Call once at process start."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = (level or os.environ.get("GD_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(level_name)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        # 20 MB x 10 backups keeps weeks of logs bounded.
        fileh = logging.handlers.RotatingFileHandler(
            run_dir / "run.log", maxBytes=20 * 1024 * 1024, backupCount=10
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)

    # Quiet noisy third-party libraries.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
