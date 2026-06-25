"""Async concurrency + request-rate limiting, per provider.

Two independent throttles compose:
* a semaphore bounding in-flight requests (concurrency), and
* a token-bucket bounding requests-per-minute (throughput).

This keeps us under provider limits during a sustained multi-week sweep without
hand-tuning sleep calls everywhere.
"""
from __future__ import annotations

import asyncio
import time


class RateLimiter:
    def __init__(self, max_concurrency: int, requests_per_minute: int = 0):
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._rpm = requests_per_minute
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def _await_slot(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait < 0:
                wait = 0.0
                self._next_slot = now
            self._next_slot += self._interval
        if wait > 0:
            await asyncio.sleep(wait)

    class _Ctx:
        def __init__(self, parent: "RateLimiter"):
            self._p = parent

        async def __aenter__(self):
            await self._p._sem.acquire()
            try:
                await self._p._await_slot()
            except BaseException:
                self._p._sem.release()
                raise
            return self

        async def __aexit__(self, *exc):
            self._p._sem.release()
            return False

    def slot(self) -> "RateLimiter._Ctx":
        return RateLimiter._Ctx(self)
