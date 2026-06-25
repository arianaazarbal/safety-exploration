"""JSONL read/write helpers and a content-addressed on-disk cache.

Every expensive call (model generation, judge scoring) is cached keyed by a hash
of its inputs, so reruns are cheap and crashed sweeps resume where they stopped.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_hash(obj: Any) -> str:
    """Deterministic hash of any JSON-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


class JsonCache:
    """A simple sharded JSON cache: one file per namespace, key -> value.

    Thread-safe for the concurrent API-call use-case. Values are flushed to disk
    on every set (cheap for our volumes; guarantees resumability after a crash).
    """

    def __init__(self, cache_dir: str | Path, namespace: str):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / f"{namespace}.json"
        self._lock = threading.Lock()
        self._store: dict[str, Any] = {}
        if self.file.exists():
            with open(self.file) as f:
                self._store = json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            tmp = self.file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self._store, f, ensure_ascii=False)
            tmp.replace(self.file)
