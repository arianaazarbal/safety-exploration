"""OpenAI-compatible backend (OpenRouter). The paper accesses closed-source
targets and the secondary judge (gpt-5-mini) through OpenRouter; we expose it as
an option for Gemini/Gemma-via-API and for the reliability cross-check judge."""
from __future__ import annotations

import os

from .base import ChatModel, GenConfig, Message


class OpenRouterModel(ChatModel):
    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        from openai import OpenAI

        self.model_id = model_id
        self.name = name or model_id
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=key, base_url=base_url)

    def supports_prefill(self) -> bool:
        # Some OpenRouter models honour a trailing assistant message as a prefill,
        # but it is provider-dependent; we do not rely on it.
        return False

    def generate(
        self, messages: list[Message], cfg: GenConfig, prefill: str | None = None
    ) -> str:
        msgs = list(messages)
        if prefill is not None:
            msgs = msgs + [{"role": "assistant", "content": prefill}]

        extra_body: dict = {}
        if cfg.disable_thinking:
            # OpenRouter reasoning toggle (ignored by models without reasoning).
            extra_body["reasoning"] = {"enabled": False}

        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=msgs,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            extra_body=extra_body or None,
        )
        return resp.choices[0].message.content or ""
