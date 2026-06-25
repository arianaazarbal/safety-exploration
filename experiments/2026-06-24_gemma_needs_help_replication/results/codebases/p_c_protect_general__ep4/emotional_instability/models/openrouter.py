"""Gemini target models via OpenRouter (OpenAI-compatible Chat Completions API).

Per Appendix B.1 the paper reaches Gemini through OpenRouter
(google/gemini-2.5-flash, google/gemini-2.5-pro) and sets thinking to false via
the API. Gemini-2.5-Pro may still emit hidden reasoning that this flag cannot
suppress (noted in the paper); we pass the documented reasoning-disable knobs
best-effort.

This is a non-Anthropic provider, used deliberately because Gemini is one of the
*target* models under study.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..config import ApiConfig
from .base import ChatMessage, ChatModel, Generation


@dataclass
class OpenRouterModel(ChatModel):
    name: str
    model_id: str            # e.g. "google/gemini-2.5-flash"
    disable_thinking: bool = True
    max_retries: int = 4
    _client: object = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        cfg = ApiConfig()
        if not cfg.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for Gemini target models."
            )
        self._client = OpenAI(
            api_key=cfg.openrouter_api_key, base_url=cfg.openrouter_base_url
        )

    def _to_openai(self, messages: Sequence[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _extra_body(self) -> dict:
        # OpenRouter exposes a unified `reasoning` control; disabling it is the
        # closest analogue to the paper's "thinking=false". For Gemini this maps
        # to minimal/no thinking where the provider honours it.
        if not self.disable_thinking:
            return {}
        return {"reasoning": {"enabled": False}}

    def generate(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> Generation:
        self._ensure_client()
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=self._to_openai(messages),
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    extra_body=self._extra_body(),
                )
                choice = resp.choices[0]
                return Generation(
                    text=(choice.message.content or "").strip(),
                    finish_reason=choice.finish_reason,
                )
            except Exception as e:  # transient API errors -> backoff
                last_err = e
                _sleep_backoff(attempt)
        raise RuntimeError(f"OpenRouter generation failed: {last_err}")

    def generate_with_prefill(self, *args, **kwargs) -> Generation:
        raise NotImplementedError(
            "Prefilling is not supported for API target models; the Section 3 "
            "base-vs-instruct experiment is run on local Gemma checkpoints only "
            "(Gemini base models are not public — see the paper's limitations)."
        )


def _sleep_backoff(attempt: int) -> None:
    import random
    import time

    time.sleep(min(2 ** attempt + random.uniform(0, 1), 30))
