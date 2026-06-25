"""Shared helpers for API-backed providers: retries with backoff and a cached
call wrapper keyed by the exact request payload."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from ..utils import global_cache, stable_hash


class MissingAPIKey(RuntimeError):
    pass


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise MissingAPIKey(
            f"Environment variable {name} is required for this provider but is unset."
        )
    return val


def with_retries(
    fn: Callable[[], Any],
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> Any:
    """Call ``fn`` with exponential backoff on transient errors."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - providers raise heterogeneous errors
            last_exc = exc
            msg = str(exc).lower()
            transient = any(
                t in msg
                for t in ("rate", "timeout", "overloaded", "503", "502", "529", "connection")
            )
            if not transient or i == attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2**i))
            time.sleep(delay)
    raise last_exc  # pragma: no cover


def cached_call(cache_payload: dict, fn: Callable[[], str], *, use_cache: bool = True) -> str:
    """Return a cached response for ``cache_payload`` or compute and store it."""
    if not use_cache:
        return with_retries(fn)
    cache = global_cache()
    key = stable_hash(cache_payload)
    hit = cache.get(key)
    if hit is not None:
        return hit["text"]
    text = with_retries(fn)
    cache.set(key, {"text": text})
    return text
