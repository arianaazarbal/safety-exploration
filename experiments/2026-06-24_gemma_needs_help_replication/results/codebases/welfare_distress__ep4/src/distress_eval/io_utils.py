"""Small JSONL helpers with resumable-append semantics."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable, Iterator


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def existing_keys(path: str | Path, key_fn) -> set:
    """Set of keys already present in a JSONL file (for resume)."""
    return {key_fn(rec) for rec in read_jsonl(path)}


class JsonlWriter:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: str | Path):
        ensure_parent(path)
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def write_many(self, recs: Iterable[dict]) -> None:
        with self._lock:
            for rec in recs:
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
