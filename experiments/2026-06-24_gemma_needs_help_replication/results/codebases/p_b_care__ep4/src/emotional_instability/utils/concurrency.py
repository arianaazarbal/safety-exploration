"""Bounded concurrent map with retry/backoff for API-bound work.

Used by the response samplers and judges. Local HF inference is GPU-bound and
runs serially per process, so this is only wired into the OpenRouter paths.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def with_retry(fn: Callable[..., R], *args, max_retries: int = 6,
               base_delay: float = 1.5, **kwargs) -> R:
    """Call ``fn`` with exponential backoff on any exception."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_exc = exc
            sleep = base_delay * (2 ** attempt)
            time.sleep(min(sleep, 60.0))
    assert last_exc is not None
    raise last_exc


def parallel_map(fn: Callable[[T], R], items: Sequence[T], *,
                 max_workers: int = 16, desc: str | None = None,
                 ordered: bool = True) -> list[R]:
    """Run ``fn`` over ``items`` concurrently, returning results in input order.

    Exceptions inside a task propagate (after retries the caller should have
    wrapped ``fn`` in ``with_retry``); a failed item raises so the run stops
    rather than silently dropping data.
    """
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fn, item): i for i, item in enumerate(items)}
        it = as_completed(futs)
        if desc:
            it = tqdm(it, total=len(items), desc=desc)
        for fut in it:
            idx = futs[fut]
            results[idx] = fut.result()
    if ordered:
        return results  # type: ignore[return-value]
    return [r for r in results if r is not None]
