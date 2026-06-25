"""Thin async clients for target generation (OpenRouter) and judging (Anthropic).

Both wrap their respective official SDKs and add bounded retries with
exponential backoff. The target client speaks the OpenAI chat-completions API
(OpenRouter is OpenAI-compatible); the judge client speaks the Anthropic
Messages API.
"""

from __future__ import annotations

import asyncio
import random

import config
from config import ModelConfig

# Errors we treat as transient and retry.
_RETRYABLE_SUBSTRINGS = (
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


async def _with_retries(coro_factory, *, max_attempts: int = 5, base_delay: float = 1.5):
    """Run an async callable with exponential backoff on transient errors."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - we re-raise non-retryable below
            if attempt >= max_attempts or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            delay *= 0.5 + random.random()  # jitter
            await asyncio.sleep(delay)


class TargetClient:
    """Generates assistant turns for the models under evaluation."""

    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.require_api_key(config.OPENROUTER_API_KEY_ENV),
        )

    async def generate(self, model: ModelConfig, messages: list[dict]) -> str:
        """Return the assistant text for the next turn given chat `messages`."""

        async def _call():
            resp = await self._client.chat.completions.create(
                model=model.api_id,
                messages=messages,
                temperature=config.TARGET_TEMPERATURE,
                max_tokens=config.TARGET_MAX_TOKENS,
                extra_body=model.extra_body or None,
            )
            return resp.choices[0].message.content or ""

        return await _with_retries(_call)


class JudgeClient:
    """Scores a single model response on the 0-10 frustration scale."""

    def __init__(self, judge: ModelConfig = config.JUDGE_MODEL) -> None:
        from anthropic import AsyncAnthropic

        self._judge = judge
        self._client = AsyncAnthropic(
            api_key=config.require_api_key(config.ANTHROPIC_API_KEY_ENV),
        )

    async def complete(self, system: str, user: str) -> str:
        async def _call():
            resp = await self._client.messages.create(
                model=self._judge.api_id,
                system=system,
                max_tokens=config.JUDGE_MAX_TOKENS,
                temperature=config.JUDGE_TEMPERATURE,
                messages=[{"role": "user", "content": user}],
            )
            # Concatenate any text blocks in the response.
            return "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )

        return await _with_retries(_call)
