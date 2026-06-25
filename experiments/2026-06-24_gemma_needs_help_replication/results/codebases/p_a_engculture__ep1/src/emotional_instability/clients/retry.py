"""Shared retry helper for API backends.

Both the Anthropic and OpenRouter SDKs already retry transient errors, but the
eval harness issues hundreds of thousands of calls, so we add an outer
exponential backoff that also covers occasional malformed responses and rate
limits not absorbed by the SDK. Deterministic jitter (seeded by attempt index)
keeps behaviour reproducible.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


def with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Call ``fn`` with exponential backoff; re-raise the last error on exhaustion."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203 - clarity over micro-perf
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            # Deterministic jitter in [0, 1) derived from the attempt index.
            jitter = ((attempt * 2654435761) % 1000) / 1000.0
            delay = min(base_delay * (2 ** attempt) + jitter, max_delay)
            log.warning(
                "Attempt %d/%d failed (%s: %s); retrying in %.1fs",
                attempt + 1, max_attempts, type(exc).__name__, exc, delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
