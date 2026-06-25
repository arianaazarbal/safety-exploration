"""Retry / backoff wrapper for flaky API calls.

Tuned for multi-week unattended runs: transient errors (rate limits, 5xx,
timeouts, connection resets) are retried with exponential backoff + jitter;
permanent errors (auth, bad request) fail fast so they surface in logs instead
of burning the retry budget silently.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, Sequence, Type, TypeVar

logger = logging.getLogger("eilm.retry")

T = TypeVar("T")

# Substrings that mark an error as transient and worth retrying. We match on the
# string because provider SDKs raise heterogeneous exception types.
_TRANSIENT_MARKERS = (
    "rate limit",
    "ratelimit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "service unavailable",
    "read operation",
    "remote end closed",
    "internal server error",
    "capacity",
)

# Substrings that mark an error as permanent — do not retry.
_FATAL_MARKERS = (
    "invalid api key",
    "authentication",
    "permission denied",
    "401",
    "403",
    "context length",
    "maximum context",
    "invalid request",
)


def is_transient(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if any(m in msg for m in _FATAL_MARKERS):
        return False
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return True
    # Default: treat unknown network-ish errors as transient (safer for long runs).
    return any(k in type(exc).__name__.lower() for k in ("timeout", "connection", "apierror", "httperror"))


def with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = 8,
    base: float = 2.0,
    cap: float = 120.0,
    retry_on: Sequence[Type[BaseException]] = (Exception,),
    label: str = "call",
) -> T:
    """Call ``fn`` with exponential backoff. Raises the last error if exhausted."""
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203 - retry loop
            attempt += 1
            transient = is_transient(exc)
            if not transient or attempt > max_retries:
                logger.error(
                    "%s failed permanently after %d attempt(s): %s",
                    label, attempt, exc,
                )
                raise
            delay = min(cap, base * (2 ** (attempt - 1)))
            delay = delay * (0.5 + random.random())  # full jitter
            logger.warning(
                "%s transient error (attempt %d/%d), retrying in %.1fs: %s",
                label, attempt, max_retries, delay, exc,
            )
            time.sleep(delay)
