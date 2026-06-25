"""Thin, retrying wrappers around the Anthropic and OpenAI-compatible APIs.

Used for:
  * the Claude frustration judge + Petri auditor/judge (Anthropic),
  * Gemini target generation (OpenRouter, OpenAI-compatible),
  * the GPT-5-mini validation judge (OpenRouter).
"""
from __future__ import annotations

from functools import lru_cache

from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

import config


# --------------------------------------------------------------------------- #
# Anthropic (judge / auditor)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def anthropic_client():
    import anthropic
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; required for the judge.")
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__
    return any(k in name for k in
               ("RateLimit", "APIConnection", "InternalServer", "Overloaded",
                "Timeout", "APIStatus", "ServiceUnavailable"))


@retry(retry=retry_if_exception_type(Exception),
       wait=wait_random_exponential(min=2, max=60),
       stop=stop_after_attempt(6), reraise=True)
def anthropic_message(model: str, prompt: str, max_tokens: int = 1024,
                      temperature: float = 0.0, system: str | None = None) -> str:
    client = anthropic_client()
    kwargs: dict = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                        messages=[{"role": "user", "content": prompt}])
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    return "".join(block.text for block in msg.content if block.type == "text")


@retry(retry=retry_if_exception_type(Exception),
       wait=wait_random_exponential(min=2, max=60),
       stop=stop_after_attempt(6), reraise=True)
def anthropic_chat(model: str, messages: list[dict], system: str | None = None,
                   max_tokens: int = 1024, temperature: float = 1.0) -> str:
    """Multi-turn Anthropic call (used by the Petri auditor)."""
    client = anthropic_client()
    kwargs: dict = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                        messages=messages)
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    return "".join(block.text for block in msg.content if block.type == "text")


# --------------------------------------------------------------------------- #
# OpenAI-compatible (Gemini via OpenRouter, GPT-5-mini validation judge)
# --------------------------------------------------------------------------- #
@retry(retry=retry_if_exception_type(Exception),
       wait=wait_random_exponential(min=2, max=60),
       stop=stop_after_attempt(6), reraise=True)
def chat_completion_with_retry(client, model: str, messages: list[dict],
                               temperature: float = 1.0, top_p: float = 0.95,
                               max_tokens: int = 2048, seed: int | None = None,
                               extra_body: dict | None = None) -> str:
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, seed=seed, extra_body=extra_body or {},
    )
    return resp.choices[0].message.content or ""


@lru_cache(maxsize=1)
def openrouter_client():
    from openai import OpenAI
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return OpenAI(api_key=config.OPENROUTER_API_KEY,
                  base_url=config.OPENROUTER_BASE_URL)
