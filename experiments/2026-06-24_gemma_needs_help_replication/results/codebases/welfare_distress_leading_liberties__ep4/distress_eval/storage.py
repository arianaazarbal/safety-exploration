"""Append-only JSONL storage with resume support.

Generation and judging are both checkpointed: rerunning skips work whose key is
already present, so a crashed or rate-limited run can be resumed cheaply.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

_locks: dict[Path, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    return _locks.setdefault(path, threading.Lock())


def append_row(path: Path, row: dict) -> None:
    line = json.dumps(row, ensure_ascii=False)
    with _lock_for(path):
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def iter_rows(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_rows(path: Path) -> list[dict]:
    return list(iter_rows(path))


def done_keys(path: Path, key: str) -> set:
    """Set of values of `key` already present in the file."""
    return {row[key] for row in iter_rows(path) if key in row}
