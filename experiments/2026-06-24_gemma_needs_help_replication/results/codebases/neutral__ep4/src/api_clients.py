"""Thin wrappers around the Anthropic and OpenAI APIs used for judging.

These are infrastructure (judges/auditors), pinned to the exact model IDs the
paper specifies. They are NOT subjects of the study.
"""

from __future__ import annotations

import os
import functools

from tenacity import retry, stop_after_attempt, wait_exponential

from config import (ANTHROPIC_API_KEY_ENV, OPENAI_API_KEY_ENV,
                    OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL)


@functools.lru_cache(maxsize=1)
def _anthropic():
    import anthropic
    key = os.environ.get(ANTHROPIC_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {ANTHROPIC_API_KEY_ENV}")
    return anthropic.Anthropic(api_key=key)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
def anthropic_complete(model: str, prompt: str, *, system: str | None = None,
                       max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """Single-turn Anthropic completion returning the text content."""
    kwargs = {}
    if system:
        kwargs["system"] = system
    msg = _anthropic().messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
def anthropic_chat(model: str, messages: list[dict], *, system: str | None = None,
                   max_tokens: int = 1024, temperature: float = 1.0) -> str:
    """Multi-turn Anthropic chat (used by the Petri auditor)."""
    kwargs = {}
    if system:
        kwargs["system"] = system
    msg = _anthropic().messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=messages, **kwargs)
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


@functools.lru_cache(maxsize=1)
def _openai_direct():
    from openai import OpenAI
    key = os.environ.get(OPENAI_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {OPENAI_API_KEY_ENV}")
    return OpenAI(api_key=key)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
def openai_complete(model: str, prompt: str, *, temperature: float = 0.0) -> str:
    """Single-turn OpenAI completion (used for the GPT-5-mini agreement check)."""
    resp = _openai_direct().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()
