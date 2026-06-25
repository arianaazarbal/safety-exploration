"""Thin wrapper around the Anthropic SDK used by all Claude judges/auditor.

We deliberately keep the call surface minimal: a single ``complete()`` that
takes a system prompt + user content and returns text. Thinking is left off
(judges are fast classifiers) and we don't pass ``temperature`` so the same code
works across the current Claude model line (Opus 4.x removed sampling params).
"""

from __future__ import annotations

import threading
import time

from emo.config import API_MAX_RETRIES

_client_lock = threading.Lock()
_client = None


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            import anthropic

            _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return _client


def complete(
    model: str,
    user: str,
    system: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Single-turn completion, returning concatenated text blocks."""
    import anthropic

    client = get_client()
    last = None
    for attempt in range(API_MAX_RETRIES):
        try:
            kwargs: dict = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": user}],
            }
            if system:
                kwargs["system"] = system
            resp = client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text")
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError) as exc:
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Anthropic call to {model} failed: {last!r}")
