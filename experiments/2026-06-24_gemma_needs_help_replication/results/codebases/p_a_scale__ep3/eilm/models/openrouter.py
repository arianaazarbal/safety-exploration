"""OpenRouter client for Gemini target models (OpenAI-compatible API).

Gemini-2.5-Flash / Pro are accessed via OpenRouter exactly as the paper does
(Appendix B.1). Thinking is disabled via provider-specific extra body params
where supported; the paper notes Gemini-2.5-Pro may still emit hidden reasoning.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..config import require_env
from ..utils.ratelimit import limiter_for
from ..utils.retry import with_retries
from .base import ChatClient, GenConfig, GenResult, Message

logger = logging.getLogger("eilm.openrouter")

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ChatClient):
    def __init__(
        self,
        api_id: str,
        name: str,
        family: str = "gemini",
        concurrency: int = 8,
        max_retries: int = 8,
        backoff_base: float = 2.0,
        backoff_max: float = 120.0,
        timeout: float = 300.0,
    ):
        from openai import OpenAI

        self.api_id = api_id
        self.name = name
        self.family = family
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._limiter = limiter_for("openrouter", concurrency)
        self._client = OpenAI(
            base_url=_BASE_URL,
            api_key=require_env("OPENROUTER_API_KEY"),
            timeout=timeout,
            max_retries=0,  # we manage retries ourselves
        )

    def _extra_body(self, cfg: GenConfig) -> dict:
        """Provider-specific params go through the OpenAI SDK's `extra_body`,
        which OpenRouter forwards as top-level request fields. A disabled
        reasoning block turns off visible thinking; the paper notes Gemini-2.5-Pro
        may still emit hidden reasoning regardless."""
        if not cfg.disable_thinking:
            return {}
        return {
            "extra_body": {
                "reasoning": {"enabled": False},
            }
        }

    def chat(self, messages: List[Message], cfg: GenConfig) -> GenResult:
        def _call() -> GenResult:
            with self._limiter.slot():
                resp = self._client.chat.completions.create(
                    model=self.api_id,
                    messages=messages,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_new_tokens,
                    seed=cfg.seed,
                    stop=cfg.stop,
                    **self._extra_body(cfg),
                )
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = {}
            if resp.usage is not None:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                }
            return GenResult(
                text=text,
                finish_reason=choice.finish_reason or "stop",
                usage=usage,
            )

        return with_retries(
            _call,
            max_retries=self._max_retries,
            base=self._backoff_base,
            cap=self._backoff_max,
            label=f"openrouter:{self.name}",
        )
