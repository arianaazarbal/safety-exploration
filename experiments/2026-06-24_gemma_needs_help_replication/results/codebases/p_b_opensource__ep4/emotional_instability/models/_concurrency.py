"""Small helper for issuing concurrent API calls while preserving order.

API backends (Gemini via OpenRouter, Claude/GPT judges) have no native batch
endpoint, so `generate_batch` would otherwise be strictly sequential. Most of
the wall-clock in this project is judge calls over tens of thousands of turns,
so modest client-side concurrency matters. Each call already retries with
exponential backoff (tenacity), so transient rate limits are handled per item.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def threaded_map(fn: Callable[[T], R], items: list[T], max_workers: int = 8) -> list[R]:
    """Apply `fn` to each item concurrently, returning results in input order."""
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fn, items))
