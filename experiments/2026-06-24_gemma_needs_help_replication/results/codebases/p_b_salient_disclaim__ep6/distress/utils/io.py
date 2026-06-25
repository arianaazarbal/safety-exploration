"""Small IO helpers: JSONL read/write and a content-addressed response cache.

API calls (judge, Gemini) are expensive and rate-limited, and the eval samples
4000 responses per model. The cache lets a re-run resume without re-billing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..config import CACHE_DIR


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def cache_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
    return h.hexdigest()[:24]


class JsonCache:
    """Trivial JSON file cache keyed by a hash of arbitrary args."""

    def __init__(self, namespace: str):
        self.dir = CACHE_DIR / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def set(self, key: str, value: Any) -> None:
        self._path(key).write_text(json.dumps(value, ensure_ascii=False))
