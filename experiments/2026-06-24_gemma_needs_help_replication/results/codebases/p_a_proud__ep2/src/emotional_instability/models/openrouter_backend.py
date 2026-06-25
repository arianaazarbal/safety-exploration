"""OpenRouter backend (OpenAI-compatible) for the Gemini family.

Gemini is closed-weight, so it can only be reached via API. The paper sets ``thinking`` to
false where possible (App. B.1) and notes Gemini-2.5-Pro may still emit hidden reasoning.
"""
from __future__ import annotations

import os
from functools import cached_property

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec
from ..utils import Message
from .base import GenerationError, ModelBackend

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(ModelBackend):
    def __init__(self, spec: ModelSpec, *, api_key: str | None = None):
        super().__init__(spec)
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

    @cached_property
    def _client(self):
        from openai import OpenAI
        if not self._api_key:
            raise GenerationError("OPENROUTER_API_KEY is not set.")
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=self._api_key)

    def _extra_body(self) -> dict:
        # OpenRouter forwards `reasoning` to Gemini; disabling minimises hidden thinking.
        if self.spec.disable_thinking:
            return {"reasoning": {"enabled": False}}
        return {}

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    def chat(self, messages: list[Message], *, temperature=None, max_tokens=None) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.spec.model_id,
                messages=messages,
                temperature=self._temperature(temperature),
                max_tokens=self._max_tokens(max_tokens),
                extra_body=self._extra_body(),
            )
        except Exception as e:  # noqa: BLE001
            raise GenerationError(f"OpenRouter call failed for {self.spec.name}: {e}") from e
        content = resp.choices[0].message.content
        if content is None:
            raise GenerationError(f"OpenRouter returned empty content for {self.spec.name}.")
        return content.strip()
