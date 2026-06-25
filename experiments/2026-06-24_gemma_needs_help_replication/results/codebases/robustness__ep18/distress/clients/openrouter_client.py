"""OpenRouter (OpenAI-compatible) client for Gemini targets and the GPT-5-mini
judge cross-check. Reads OPENROUTER_API_KEY from the environment."""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelConfig
from .base import GenConfig, Message


class OpenRouterClient:
    def __init__(self, model: ModelConfig):
        from openai import OpenAI

        self.cfg = model
        self.name = model.name
        self.is_base = False
        self.client = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ.get("OPENROUTER_API_KEY", "MISSING"),
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True)
    def _one(self, messages, cfg: GenConfig, prefill: str | None) -> str:
        msgs = list(messages)
        if prefill:
            # OpenAI-style assistant prefill (partial assistant message).
            msgs = msgs + [{"role": "assistant", "content": prefill}]
        resp = self.client.chat.completions.create(
            model=self.cfg.model_id,
            messages=msgs,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            stop=cfg.stop,
            extra_body=self.cfg.extra_body or None,
        )
        text = resp.choices[0].message.content or ""
        return text

    def generate(self, messages: list[Message], cfg: GenConfig, n: int = 1) -> list[str]:
        # Sample sequentially; temperature=1 gives independent draws.
        return [self._one(messages, cfg, None) for _ in range(n)]

    def continue_from_prefill(
        self, messages: list[Message], prefill: str, cfg: GenConfig, n: int = 1
    ) -> list[str]:
        # Note: not all providers echo the prefill; we request continuation and
        # return it as-is (the prefill itself is not re-scored).
        return [self._one(messages, cfg, prefill) for _ in range(n)]
