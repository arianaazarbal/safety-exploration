"""Small helpers shared by LLM-judge code: robust JSON extraction and retry."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

# Curly-quote characters that PDF extraction (and some models) emit instead of
# ASCII quotes; normalising them lets json.loads succeed.
_SMART_QUOTES = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "„": '"', "″": '"',
}


def _normalise_quotes(s: str) -> str:
    for bad, good in _SMART_QUOTES.items():
        s = s.replace(bad, good)
    return s


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract the last balanced ``{...}`` object from a model response.

    Judges are instructed to end with JSON; models sometimes wrap it in prose or
    fenced code blocks, so we scan for the final balanced brace span.
    """
    if not text:
        return None
    text = _normalise_quotes(text)

    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fence:
        candidates.append(fence.group(1))

    # Find all balanced top-level brace spans.
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])

    # Try the last candidate first (judges put JSON at the end).
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Tolerate a trailing comma before a closing brace/bracket.
            patched = re.sub(r",(\s*[}\]])", r"\1", cand)
            try:
                return json.loads(patched)
            except json.JSONDecodeError:
                continue
    return None


def with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = 4,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
    on_error: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """Call ``fn`` with exponential backoff. Re-raises after the last attempt."""
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except exceptions as exc:  # noqa: BLE001 - deliberate broad retry
            last = exc
            if on_error:
                on_error(attempt, exc)
            if attempt == max_retries - 1:
                break
            time.sleep(base_delay * (2 ** attempt))
    assert last is not None
    raise last
