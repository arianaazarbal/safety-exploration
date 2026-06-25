"""Bounded-concurrency map for I/O-bound API calls (judging, Petri, Gemini).

Local-GPU generation is *not* run through this -- it batches on-device instead.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def thread_map(fn: Callable[[T], R], items: Iterable[T], *,
               max_workers: int = 8, ordered: bool = True) -> List[R]:
    items = list(items)
    results: List[R] = [None] * len(items)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results
