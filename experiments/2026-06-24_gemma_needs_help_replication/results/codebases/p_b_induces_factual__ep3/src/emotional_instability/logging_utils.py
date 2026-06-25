"""Shared logging + JSONL persistence helpers.

Every experiment streams its raw rollouts/scores to a JSONL file so that
analysis (metrics, figures) is decoupled from generation and re-runnable
without re-querying any model.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("EMO_LOGLEVEL", "INFO"))
        logger.propagate = False
    return logger


def write_jsonl(path: str | os.PathLike, records: Iterable[dict[str, Any]]) -> int:
    """Write records to a JSONL file, creating parent dirs. Returns count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | os.PathLike, record: dict[str, Any]) -> None:
    """Append a single record (used for incremental / resumable runs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
