"""Async OpenAI-compatible chat client used for both target models and the judge.

A single client abstraction covers all cases because every backend we use speaks
the OpenAI /chat/completions protocol:
  * Gemma / Gemini via OpenRouter (default)
  * Gemma via a local `vllm serve ...` instance (set base_url to the local URL)
  * Claude judge via OpenRouter (or an OpenAI-compatible Anthropic proxy)

Reasoning ("thinking") is disabled where requested via OpenRouter's `reasoning`
control. The paper notes Gemini-2.5-Pro and GPT-5.2 may still emit hidden
reasoning despite this; we surface that caveat in DESIGN.md.
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI
from openai import APIError, APIConnectionError, RateLimitError

from . import config


class ChatClient:
    def __init__(
        self,
        slug: str,
        base_url: str,
        api_key_env: str,
        disable_reasoning: bool = False,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            # Allow a dummy key for local vLLM servers that ignore auth.
            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.slug = slug
        self.disable_reasoning = disable_reasoning
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,  # we implement our own backoff below
        )

    def _extra_body(self) -> dict:
        body: dict = {}
        if self.disable_reasoning:
            # OpenRouter unified reasoning control. For Gemini this maps to a
            # zero/low thinking budget; for Gemma (no thinking) it is a no-op.
            body["reasoning"] = {"enabled": False}
        return body

    async def generate(
        self,
        messages: list[dict],
        temperature: float = config.TEMPERATURE,
        top_p: float = config.TOP_P,
        max_tokens: int = config.MAX_TOKENS,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.slug,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=self._extra_body(),
                )
                choice = resp.choices[0]
                content = choice.message.content
                return content if content is not None else ""
            except (RateLimitError, APIConnectionError) as exc:
                last_exc = exc
                delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
            except APIError as exc:
                last_exc = exc
                # Retry 5xx; re-raise client errors immediately.
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"generate() failed after {config.MAX_RETRIES} retries for model {self.slug}"
        ) from last_exc


def client_for_model(model_cfg) -> ChatClient:
    return ChatClient(
        slug=model_cfg.slug,
        base_url=model_cfg.base_url,
        api_key_env=model_cfg.api_key_env,
        disable_reasoning=model_cfg.disable_reasoning,
    )


def client_for_judge(judge_cfg=config.JUDGE) -> ChatClient:
    return ChatClient(
        slug=judge_cfg.slug,
        base_url=judge_cfg.base_url,
        api_key_env=judge_cfg.api_key_env,
        disable_reasoning=False,
    )
