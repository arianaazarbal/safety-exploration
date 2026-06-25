"""Small IO helpers: JSONL read/write and a simple on-disk cache for the
(expensive, non-deterministic) LLM calls so reruns are cheap and resumable."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

_WRITE_LOCK = threading.Lock()


def write_jsonl(path: str | Path, rows: Iterable[dict], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    """Thread-safe single-row append (used by concurrent rollout workers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_hash(*parts: Any) -> str:
    """Deterministic hash over the given parts, used as cache / response keys."""
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode())
    return h.hexdigest()[:16]


class JsonlCache:
    """Append-only key->value cache backed by a JSONL file.

    Used to memoise judge ratings / paraphrases so re-running an analysis does
    not re-pay for API calls. Keys are stable hashes of the call inputs.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {}
        for row in iter_jsonl(self.path):
            self._data[row["k"]] = row["v"]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value
        append_jsonl(self.path, {"k": key, "v": value})
