"""Shared exponential-backoff retry for API backends.

The official SDKs already retry rate-limits / 5xx, but the elicitation harness
issues thousands of calls and benefits from a uniform, slightly more patient
wrapper around transient failures. Deterministic jitter is keyed off the attempt
index so retries don't depend on wall-clock RNG.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    on_error: Callable[[int, Exception], None] | None = None,
) -> T:
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - backends raise heterogeneous types
            last = exc
            if on_error is not None:
                on_error(attempt, exc)
            if attempt == max_attempts - 1:
                break
            delay = min(base_delay * (2**attempt), max_delay)
            # deterministic jitter in [0, 1)
            jitter = ((attempt * 2654435761) % 1000) / 1000.0
            time.sleep(delay + jitter)
    assert last is not None
    raise last
