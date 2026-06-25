"""Durable, resumable I/O primitives.

The whole replication is designed to be killed and restarted at any time over a
multi-week run without losing or duplicating work. That guarantee lives here:

* `JsonlStore` is an append-only log keyed by an idempotent task key. On open it
  loads the set of keys already present, so callers can cheaply ask "is this
  task done?" and skip it. Writes are serialised through a lock and flushed.
* `stable_key` produces a deterministic content hash for a task spec.
* `atomic_write_*` write to a temp file and rename, so a crash mid-write never
  leaves a half-written artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator


def stable_key(*parts: Any) -> str:
    """Deterministic short hash of arbitrary JSON-serialisable parts.

    Used as the idempotent identity of a unit of work (one generation, one judge
    call, ...). Same inputs -> same key -> skipped on resume.
    """
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlStore:
    """Append-only JSONL store with idempotent keys for resumable runs.

    Each record must carry a ``key`` field (the caller supplies it, typically a
    `stable_key(...)`). Re-appending a key that already exists is a no-op, which
    is what makes the whole pipeline safe to re-run.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._keys: set[str] = set()
        self._load_keys()

    def _load_keys(self) -> None:
        for rec in read_jsonl(self.path):
            k = rec.get("key")
            if k is not None:
                self._keys.add(k)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> set[str]:
        return set(self._keys)

    def append(self, record: dict) -> bool:
        """Append a record if its key is new. Returns True if written."""
        key = record["key"]
        with self._lock:
            if key in self._keys:
                return False
            line = json.dumps(record, ensure_ascii=False, default=str)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._keys.add(key)
            return True

    def records(self) -> Iterator[dict]:
        yield from read_jsonl(self.path)

    def filter_pending(self, specs: Iterable[tuple[str, Any]]) -> list[Any]:
        """Given (key, payload) pairs, return payloads whose key is not yet done."""
        return [payload for key, payload in specs if key not in self._keys]
