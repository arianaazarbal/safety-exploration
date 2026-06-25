"""Threaded map with retry/backoff for API-bound work (judge, Gemini, Petri).

API calls dominate wall-clock for the closed-model and judge paths, so we fan
out with a thread pool. `parallel_map` preserves input order and surfaces
per-item exceptions as `None` (callers filter) unless `raise_on_error=True`.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_WORKERS = 16


def with_retries(
    fn: Callable[..., R],
    *,
    attempts: int = 6,
    max_wait: float = 60.0,
    exc_types: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[..., R]:
    """Wrap `fn` with exponential backoff. Used for rate-limited API calls."""

    return retry(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=wait_random_exponential(multiplier=1, max=max_wait),
        retry=retry_if_exception_type(exc_types),
    )(fn)


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    workers: int = DEFAULT_WORKERS,
    raise_on_error: bool = False,
    desc: str | None = None,
) -> list[R | None]:
    items = list(items)
    results: list[R | None] = [None] * len(items)
    if not items:
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                if raise_on_error:
                    raise
                log.warning("item %d (%s) failed: %s", idx, desc, exc)
                results[idx] = None
            done += 1
            if desc and done % 50 == 0:
                log.info("%s: %d/%d", desc, done, len(items))
    return results
