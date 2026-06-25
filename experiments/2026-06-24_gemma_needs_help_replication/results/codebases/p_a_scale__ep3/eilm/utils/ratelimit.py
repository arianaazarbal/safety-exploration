"""Lightweight concurrency limiter for API providers.

A bounded semaphore caps simultaneous in-flight requests per provider, which
keeps us under provider concurrency limits without a global lock. Combined with
``with_retries`` this is enough to stay healthy across a multi-week run.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict


class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int):
        self._sem = threading.BoundedSemaphore(max(1, int(max_concurrent)))

    @contextmanager
    def slot(self):
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


_LIMITERS: Dict[str, ConcurrencyLimiter] = {}
_GUARD = threading.Lock()


def limiter_for(provider: str, max_concurrent: int) -> ConcurrencyLimiter:
    with _GUARD:
        if provider not in _LIMITERS:
            _LIMITERS[provider] = ConcurrencyLimiter(max_concurrent)
        return _LIMITERS[provider]
