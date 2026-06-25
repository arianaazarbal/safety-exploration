"""Shared helpers for API-backed clients: retry/backoff and a disk cache.

The disk cache is keyed on a hash of the full request payload so that re-running
a pipeline (e.g. re-judging) is cheap and deterministic. Caching matters a lot
here because judging 4000 responses/model is the dominant API cost.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .. import config


def request_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class DiskCache:
    """Tiny thread-safe JSON-value cache sharded by key prefix."""

    def __init__(self, namespace: str):
        self.dir = config.CACHE_DIR / namespace
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            tmp = self._path(key).with_suffix(".tmp")
            tmp.write_text(json.dumps(value))
            tmp.replace(self._path(key))


def with_retries(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate an API call with exponential backoff (handles rate limits)."""
    return retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential_jitter(initial=1, max=60),
        reraise=True,
    )(fn)
