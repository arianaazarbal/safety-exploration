"""Shared retry/backoff helper for flaky API calls."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = 4,
    base_delay: float = 1.0,
    factor: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call ``fn`` with exponential backoff. Re-raises the last error if all
    attempts fail. Deterministic backoff (no jitter) to keep runs reproducible."""
    last: BaseException | None = None
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except exceptions as exc:  # noqa: BLE001 - intentional broad catch for API calls
            last = exc
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay *= factor
    assert last is not None
    raise last
