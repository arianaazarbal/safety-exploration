"""JSONL I/O and a small content-addressed cache for model/judge calls.

All experiments write append-only JSONL so long runs can be resumed: before
issuing an (expensive) API/inference call, callers check the cache keyed by a
hash of the request. This makes the 4000-responses-per-model sweeps restartable.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class JsonlWriter:
    """Thread-safe append-only writer used by concurrent samplers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def append(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def hash_key(obj: Any) -> str:
    """Stable hash of an arbitrary JSON-serialisable request."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]
