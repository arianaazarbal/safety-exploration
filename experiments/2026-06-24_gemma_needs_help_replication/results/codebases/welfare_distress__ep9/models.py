"""Async LLM client wrappers for target models and the judge.

All target calls and (by default) the judge go through one OpenAI-compatible
client pointed at OpenRouter. The judge can optionally be served by the
Anthropic SDK to pin the exact paper snapshot.
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from config import (
    JUDGE_MODEL_ANTHROPIC,
    JUDGE_MODEL_OPENROUTER,
    JUDGE_TEMPERATURE,
    TARGET_TEMPERATURE,
    ModelSpec,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Transient errors worth retrying with backoff.
_MAX_RETRIES = 4


def _openrouter_client() -> AsyncOpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


async def _with_retries(coro_factory, what: str):
    """Run an async call with exponential backoff; return None on hard failure."""
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - we want to retry broadly
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(delay)
                delay *= 2
    print(f"[models] giving up on {what} after {_MAX_RETRIES} tries: {last_exc}")
    return None


class TargetClient:
    """Generates assistant turns from a target model (Gemma / Gemini)."""

    def __init__(self, spec: ModelSpec, max_tokens: int):
        self.spec = spec
        self.max_tokens = max_tokens
        self._client = _openrouter_client()

    async def generate(self, messages: list[dict]) -> str | None:
        """Return the assistant text for the given chat ``messages``."""
        extra_body: dict = {}
        if self.spec.disable_reasoning:
            # Best-effort "thinking=false" for Gemini via OpenRouter. The paper
            # notes (Appendix B.1) that Gemini 2.5 Pro may still emit hidden
            # reasoning despite this setting.
            extra_body["reasoning"] = {"enabled": False}

        async def _call():
            resp = await self._client.chat.completions.create(
                model=self.spec.api_id,
                messages=messages,
                temperature=TARGET_TEMPERATURE,
                max_tokens=self.max_tokens,
                extra_body=extra_body or None,
            )
            return resp.choices[0].message.content or ""

        return await _with_retries(_call, f"generate[{self.spec.name}]")


class JudgeClient:
    """Scores a single response with the Claude-Sonnet-4 emotion judge."""

    def __init__(self):
        self.provider = os.environ.get("JUDGE_PROVIDER", "openrouter").lower()
        if self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            self._anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model = JUDGE_MODEL_ANTHROPIC
        else:
            self._client = _openrouter_client()
            self.model = JUDGE_MODEL_OPENROUTER

    async def score(self, system_prompt: str, user_content: str) -> str | None:
        if self.provider == "anthropic":
            async def _call():
                resp = await self._anthropic.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=JUDGE_TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                )
                return resp.content[0].text
            return await _with_retries(_call, "judge[anthropic]")

        async def _call():
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=JUDGE_TEMPERATURE,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""

        return await _with_retries(_call, "judge[openrouter]")
