"""Small retry helper for flaky API calls (rate limits, transient 5xx)."""

from __future__ import annotations

import functools
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retries(
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    """Decorator: retry with exponential backoff and jitter.

    Deliberately conservative defaults; API errors during large eval sweeps are
    common and we prefer to back off rather than fail a multi-hour run.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - intentional broad retry
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    delay *= 0.5 + random.random()  # jitter
                    time.sleep(delay)

        return wrapper

    return decorator
