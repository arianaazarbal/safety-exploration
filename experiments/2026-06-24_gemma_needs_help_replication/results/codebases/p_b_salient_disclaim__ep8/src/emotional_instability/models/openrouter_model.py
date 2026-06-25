"""OpenRouter backend (OpenAI-compatible) for Gemini-2.5-Flash / Pro.

Reads OPENROUTER_API_KEY from the environment. `disable_thinking` maps to the
provider-specific reasoning toggle the paper sets ("thinking=false via the
API"). Note the paper's caveat that Gemini-2.5-Pro may still emit hidden
reasoning regardless.
"""
from __future__ import annotations

import os
import time

from .base import GenerationConfig, Message, ModelClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(ModelClient):
    def __init__(self, spec, max_retries: int = 5):
        super().__init__(spec)
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self.api_id = spec.get("api_id")
        self.disable_thinking = bool(spec.get("disable_thinking"))
        self.max_retries = max_retries

    def _extra_body(self) -> dict:
        if self.disable_thinking:
            # OpenRouter unifies reasoning control under `reasoning`.
            # enabled=false / effort minimal disables thinking where supported.
            return {"reasoning": {"enabled": False}}
        return {}

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.api_id,
                    messages=messages,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_new_tokens,
                    seed=cfg.seed,
                    extra_body=self._extra_body(),
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - retry transient API errors
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")
