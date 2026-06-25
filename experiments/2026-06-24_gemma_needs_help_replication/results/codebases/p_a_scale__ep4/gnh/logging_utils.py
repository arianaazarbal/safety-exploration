"""Structured logging + a tiny usage/cost accumulator.

Over a multi-week unattended run we want (a) a persistent log we can tail, and
(b) a running tally of token usage so cost doesn't surprise anyone.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

_CONFIGURED = False


def setup_logging(output_dir: str | Path, level: str = "INFO", name: str = "gnh") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if _CONFIGURED:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str = "gnh") -> logging.Logger:
    return logging.getLogger(name)


class UsageTracker:
    """Thread-safe token tally, grouped by model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, dict[str, int]] = {}

    def add(self, model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._lock:
            d = self._by_model.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
            d["prompt"] += int(prompt_tokens or 0)
            d["completion"] += int(completion_tokens or 0)
            d["calls"] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {m: dict(v) for m, v in self._by_model.items()}


USAGE = UsageTracker()
