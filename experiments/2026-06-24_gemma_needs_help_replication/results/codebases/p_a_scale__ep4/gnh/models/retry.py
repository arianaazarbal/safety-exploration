"""Retry policy for transient API failures.

We retry on network errors, timeouts, 429s, and 5xx with exponential backoff +
jitter. Non-retryable client errors (400/401/403/404) raise immediately so a
misconfiguration fails fast instead of silently looping for weeks.
"""
from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

from gnh.logging_utils import get_logger

T = TypeVar("T")
log = get_logger()


class RetryableError(Exception):
    """Raised by backends for failures worth retrying."""


class FatalAPIError(Exception):
    """Raised for client errors that will never succeed on retry."""


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float = 2.0,
    max_delay: float = 90.0,
    what: str = "request",
) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except FatalAPIError:
            raise
        except RetryableError as e:
            attempt += 1
            if attempt > max_retries:
                log.error("%s failed after %d retries: %s", what, max_retries, e)
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()  # jitter in [0.5x, 1.5x]
            log.warning("%s retry %d/%d in %.1fs: %s", what, attempt, max_retries, delay, e)
            await asyncio.sleep(delay)
        except (asyncio.TimeoutError, ConnectionError) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * (0.5 + random.random())
            log.warning("%s transient retry %d/%d in %.1fs: %s", what, attempt, max_retries, delay, e)
            await asyncio.sleep(delay)
