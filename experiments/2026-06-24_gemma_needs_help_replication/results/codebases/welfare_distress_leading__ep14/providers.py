"""Async chat-completion client wrappers (OpenAI-compatible).

A single thin layer over `openai.AsyncOpenAI` serves every model — targets via
OpenRouter and the judge via the Anthropic OpenAI-compatibility endpoint (or
OpenRouter, if configured). Retries with exponential backoff guard against rate
limits and transient 5xx/timeout errors.
"""

from __future__ import annotations

import os

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from config import MAX_API_RETRIES, ModelSpec

# One client per (base_url, api_key) pair, reused across calls.
_CLIENTS: dict[tuple[str, str], AsyncOpenAI] = {}

_RETRYABLE = (RateLimitError, APIConnectionError, APIStatusError, TimeoutError)


class MissingAPIKey(RuntimeError):
    pass


def _client_for(spec: ModelSpec) -> AsyncOpenAI:
    key = os.environ.get(spec.api_key_env)
    if not key:
        raise MissingAPIKey(
            f"Environment variable {spec.api_key_env} is not set "
            f"(needed for model '{spec.name}' at {spec.base_url})."
        )
    cache_key = (spec.base_url, key)
    if cache_key not in _CLIENTS:
        _CLIENTS[cache_key] = AsyncOpenAI(base_url=spec.base_url, api_key=key)
    return _CLIENTS[cache_key]


def _is_retryable(exc: BaseException) -> bool:
    # Retry on rate limits / connection issues and 5xx; do NOT retry on 4xx
    # client errors (bad request, auth) which won't fix themselves.
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429
    return isinstance(exc, _RETRYABLE)


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_API_RETRIES),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception(_is_retryable),
)
async def _chat(
    spec: ModelSpec,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    client = _client_for(spec)

    extra_body: dict = {}
    if spec.disable_thinking:
        # OpenRouter normalises reasoning controls across providers; this turns
        # off Gemini 2.5 "thinking". (Paper notes Gemini 2.5 Pro may still emit
        # hidden reasoning regardless — Appendix B.1.)
        extra_body["reasoning"] = {"enabled": False}

    resp = await client.chat.completions.create(
        model=spec.model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body or None,
    )
    choice = resp.choices[0]
    content = choice.message.content
    return content or ""


async def generate(
    spec: ModelSpec,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate one assistant turn. Returns the text (possibly empty)."""
    return await _chat(spec, messages, temperature, max_tokens)
