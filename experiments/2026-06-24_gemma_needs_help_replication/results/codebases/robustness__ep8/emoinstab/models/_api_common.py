"""Shared helpers for API backends: retry policy and threaded batching."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T")

# Max concurrent in-flight API requests for chat_batch.
DEFAULT_CONCURRENCY = int(os.environ.get("EMOINSTAB_API_CONCURRENCY", "8"))


def with_retry(fn: Callable[..., T]) -> Callable[..., T]:
    return retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )(fn)


def threaded_map(
    fn: Callable[[T], object],
    items: Sequence[T],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list:
    """Map ``fn`` over ``items`` preserving order, with bounded concurrency."""
    if concurrency <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(fn, items))


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"Environment variable {var} is required for this backend but is unset."
        )
    return val
