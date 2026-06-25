"""OpenRouter (OpenAI-compatible) backend for the Gemini API models.

Gemini is closed-weights, so there is no prefilled-generation or token-level
access -- those methods inherit the ``NotImplementedError`` from the base class.
We disable "thinking"/reasoning where the API allows it (per Appendix B.1: "we
set thinking to be false via the API. However, Gemini-2.5 Pro ... may produce
hidden reasoning that is not prevented by this setting").
"""
from __future__ import annotations

import time

import config
from .base import ChatClient, GenConfig, Message


class OpenRouterClient(ChatClient):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        from openai import OpenAI

        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )

    def _extra_body(self) -> dict:
        # OpenRouter passes provider-specific knobs through `extra_body`.
        # For Gemini, request zero reasoning tokens to disable thinking.
        return {"reasoning": {"max_tokens": 0, "enabled": False}}

    def generate(self, messages: list[Message], cfg: GenConfig) -> list[str]:
        outputs: list[str] = []
        # OpenRouter/Gemini does not reliably honour n>1, so we loop.
        for _ in range(cfg.n):
            text = self._one_call(messages, cfg)
            outputs.append(text)
        return outputs

    def _one_call(self, messages: list[Message], cfg: GenConfig,
                  max_retries: int = 5) -> str:
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_new_tokens,
                    stop=list(cfg.stop) if cfg.stop else None,
                    extra_body=self._extra_body(),
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - API hiccups, retry
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""  # unreachable
