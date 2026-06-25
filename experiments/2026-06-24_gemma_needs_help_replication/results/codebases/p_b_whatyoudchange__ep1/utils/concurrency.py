"""Bounded-concurrency map with retry/backoff for API-bound work."""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from tqdm import tqdm

from config import API_MAX_RETRIES, API_MAX_WORKERS

T = TypeVar("T")
R = TypeVar("R")


def with_retry(fn: Callable[..., R], *args, max_retries: int = API_MAX_RETRIES,
               base_delay: float = 1.0, max_delay: float = 30.0, **kwargs) -> R:
    """Call `fn` with exponential backoff + jitter. Re-raises after `max_retries`."""
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:        # noqa: BLE001 - API clients raise many types
            last = e
            delay = min(base_delay * (2 ** attempt), max_delay)
            time.sleep(delay + random.uniform(0, 0.5 * delay))
    assert last is not None
    raise last


def parallel_map(fn: Callable[[T], R], items: Iterable[T], *,
                 max_workers: int = API_MAX_WORKERS, desc: str | None = None,
                 ) -> list[R]:
    """Order-preserving threaded map. Each item runs `fn` (wrap in with_retry as
    needed). Exceptions propagate after the pool drains."""
    items = list(items)
    results: list[R | None] = [None] * len(items)
    errors: list[tuple[int, Exception]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:    # noqa: BLE001
                errors.append((i, e))
    if errors:
        i, e = errors[0]
        raise RuntimeError(f"{len(errors)} task(s) failed; first at index {i}: {e}") from e
    return results  # type: ignore[return-value]
