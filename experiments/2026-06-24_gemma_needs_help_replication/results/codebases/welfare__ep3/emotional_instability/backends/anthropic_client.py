"""Thin Anthropic-SDK wrapper used by the LLM judge (Section 2) and the Petri
auditor/judge (Section 4). Kept separate from target-model backends because the
judge always runs on Claude regardless of which target is being evaluated.
"""
from __future__ import annotations

from typing import Optional

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import ANTHROPIC_API_KEY

_CLIENT: Optional[anthropic.Anthropic] = None


def _client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot run the judge.")
        _CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _CLIENT


@retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError)
    ),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)
def complete(
    model: str,
    system: Optional[str],
    messages: list[dict],
    max_tokens: int = 1024,
) -> str:
    """Single-turn (or multi-turn) Claude completion returning concatenated text.

    The paper's judge models (claude-sonnet-4-20250514 / claude-opus-4-20250514)
    use the classic Messages API surface; we do not request adaptive thinking or
    structured outputs so the call works identically on those exact model IDs.
    """
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        kwargs["system"] = system
    resp = _client().messages.create(**kwargs)
    return "".join(b.text for b in resp.content if b.type == "text")
