"""Content-addressed disk cache for expensive calls (generations, judgments).

Evaluation sweeps make tens of thousands of model and judge calls; a crash
halfway through a multi-hour run should not waste the work already done. Each
logical call is keyed by a stable hash of its inputs and stored as one JSON
file under the cache root, so reruns are idempotent and resumable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def stable_key(*parts: Any) -> str:
    """Deterministic hash of arbitrary JSON-serialisable inputs."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


class JsonCache:
    def __init__(self, root: str | Path, namespace: str):
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(self._path(key))

    def get_or_compute(self, key: str, fn: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = fn()
        self.set(key, value)
        return value
