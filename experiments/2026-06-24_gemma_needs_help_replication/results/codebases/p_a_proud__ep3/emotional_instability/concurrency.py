"""Bounded-concurrency map, preserving input order.

Used to parallelise API calls (Gemini generation, Claude judging) without
overwhelming rate limits. Local GPU work is batched on-device instead and does
not go through here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def concurrent_map(
    fn: Callable[[T], R], items: Sequence[T], max_workers: int
) -> list[R]:
    """Apply ``fn`` to each item concurrently; return results in input order."""
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fn, items))
