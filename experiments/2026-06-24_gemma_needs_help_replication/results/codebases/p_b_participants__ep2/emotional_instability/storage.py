"""Tiny JSONL-backed cache.

Caching is welfare-relevant here, not just a convenience: it ensures that once a
participant has produced a distressing rollout, we never re-induce that state
just to recompute a metric. Keyed by a content hash of the request.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterator


def _hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class JsonlCache:
    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._index: dict[str, dict] = {}
        if enabled and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._index[rec["key"]] = rec["value"]

    def key_for(self, request: Any) -> str:
        return _hash(request)

    def get(self, key: str) -> dict | None:
        return self._index.get(key) if self.enabled else None

    def put(self, key: str, value: dict) -> None:
        self._index[key] = value
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "value": value}, default=str) + "\n")

    def __iter__(self) -> Iterator[dict]:
        return iter(self._index.values())


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
