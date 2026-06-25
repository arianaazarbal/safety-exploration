"""Small JSONL helpers shared by the runner, scorer, and analysis."""

from __future__ import annotations

import json
import os
import threading
from typing import Iterator


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlWriter:
    """Thread-safe append-only JSONL writer (line-buffered, flushed per write).

    Append-only + flush-per-line means a crashed run leaves a valid prefix on
    disk, which the resumability logic in the runner/scorer relies on.
    """

    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path) or ".")
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
