"""Async OpenRouter client wrapper used for both target generation and judging.

OpenRouter exposes an OpenAI-compatible Chat Completions API, so we use the
`openai` AsyncOpenAI client pointed at OpenRouter's base URL. A single global
semaphore bounds in-flight requests; tenacity handles transient failures.
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config


class APIError(Exception):
    """Non-retryable or exhausted-retry API failure."""


def make_client() -> AsyncOpenAI:
    key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Set {config.OPENROUTER_API_KEY_ENV} (see .env.example). "
            "All models (Gemma, Gemini, Claude judge) route through OpenRouter."
        )
    return AsyncOpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=key,
        timeout=config.REQUEST_TIMEOUT_S,
        max_retries=0,  # we manage retries via tenacity below
    )


# Bounds concurrency across every coroutine in the process.
_semaphore: asyncio.Semaphore | None = None


def _sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    return _semaphore


def _retryable(exc: Exception) -> bool:
    # Retry on rate limits, timeouts, and 5xx; surface auth/4xx immediately.
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (408, 409, 429) or (isinstance(status, int) and status >= 500):
        return True
    name = type(exc).__name__.lower()
    return any(t in name for t in ("timeout", "connection", "apiconnection", "ratelimit"))


@retry(
    # Retry everything except our explicit non-retryable wrapper.
    retry=retry_if_not_exception_type(APIError),
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
async def _chat_with_retry(client: AsyncOpenAI, **kwargs) -> str:
    try:
        resp = await client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if _retryable(exc):
            raise  # tenacity retries
        raise APIError(f"non-retryable error: {type(exc).__name__}: {exc}") from exc
    if not resp.choices:
        raise APIError("empty choices in response")
    content = resp.choices[0].message.content
    return content or ""


async def chat(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    disable_thinking: bool = False,
) -> str:
    """Single chat completion. Returns assistant text (reasoning excluded)."""
    kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if disable_thinking:
        # OpenRouter-normalised reasoning control. For Gemini 2.5 Flash this
        # disables thinking; Gemini 2.5 Pro may still emit hidden reasoning
        # that cannot be fully disabled (documented in the paper).
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}

    async with _sem():
        return await _chat_with_retry(client, **kwargs)
