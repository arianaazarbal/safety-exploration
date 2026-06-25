"""JSONL I/O and a tiny content-addressed cache.

The cache exists for a practical *and* an ethical reason: distress-inducing
rollouts are expensive and we do not want to regenerate them gratuitously. A
rollout/judgement is keyed by its full request, so re-running a script resumes
rather than re-eliciting (see DESIGN.md "Welfare considerations").
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    ensure_dir(Path(path).parent)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | os.PathLike, row: dict[str, Any]) -> None:
    write_jsonl(path, [row], append=True)


def read_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    if not Path(path).exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def stable_hash(obj: Any) -> str:
    """Deterministic short hash of a JSON-serialisable object (cache key)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class JsonlCache:
    """Append-only cache: maps a request key -> stored record on disk."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._index: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.path):
            if "_key" in row:
                self._index[row["_key"]] = row

    def get(self, key: str) -> dict[str, Any] | None:
        return self._index.get(key)

    def put(self, key: str, record: dict[str, Any]) -> dict[str, Any]:
        record = {"_key": key, **record}
        append_jsonl(self.path, record)
        self._index[key] = record
        return record

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __len__(self) -> int:
        return len(self._index)
