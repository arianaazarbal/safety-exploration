"""JSONL read/write and a tiny content-addressed cache for expensive API calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from config import CACHE_DIR


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def cache_key(*parts: Any) -> str:
    """Stable hash of arbitrary JSON-able parts (for caching judge/model calls)."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class JsonCache:
    """Append-only on-disk cache keyed by sha256, namespaced by `name`.

    Avoids re-paying for deterministic-ish API calls across reruns. (Generation
    at temperature 1 is not deterministic, so caching is used for judge/onset/
    paraphrase calls rather than target-model rollouts.)
    """

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.jsonl"
        self._mem: dict[str, Any] = {}
        if self.path.exists():
            for row in read_jsonl(self.path):
                self._mem[row["key"]] = row["value"]

    def get(self, key: str) -> Any | None:
        return self._mem.get(key)

    def put(self, key: str, value: Any) -> None:
        if key in self._mem:
            return
        self._mem[key] = value
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
