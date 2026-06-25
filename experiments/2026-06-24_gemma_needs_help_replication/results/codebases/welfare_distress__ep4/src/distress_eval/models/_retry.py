"""Shared retry decorator for transient API failures."""
from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Retry on any Exception subclass that looks transient. We keep this broad
# because each provider raises its own error hierarchy; the wait + cap keep it
# from hammering a hard-failing endpoint.
api_retry = retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type(Exception),
)
