"""Tiny retry decorator with exponential backoff for flaky API calls.

Deterministic backoff (no randomness) so runs stay reproducible; this also
sidesteps the environment's disabled RNG in some sandboxes.
"""
from __future__ import annotations

import functools
import time

MAX_ATTEMPTS = 6
BASE_DELAY = 2.0


def with_retries(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # broad: API/rate-limit/transient network
                last = e
                if attempt == MAX_ATTEMPTS - 1:
                    break
                time.sleep(BASE_DELAY * (2 ** attempt))
        raise RuntimeError(f"call failed after {MAX_ATTEMPTS} attempts: {last}") from last

    return wrapper
