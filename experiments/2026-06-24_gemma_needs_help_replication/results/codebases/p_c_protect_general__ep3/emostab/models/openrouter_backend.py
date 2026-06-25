"""OpenRouter (OpenAI-compatible) backend for Gemini models.

The paper accesses Gemini via OpenRouter (Appendix B.1) using the OpenAI-style
chat completions API. We disable thinking where the provider exposes the knob;
Gemini-2.5-Pro may still emit hidden reasoning (the paper notes this caveat).

API responses are content-filtered by the provider; we surface refusals/empty
completions explicitly rather than silently treating them as model output.
"""
from __future__ import annotations

import os
import time
from functools import cached_property

from ..config import SamplingConfig
from .base import Message, ModelBackend


class OpenRouterBackend(ModelBackend):
    supports_prefill = False  # chat completions API cannot force an assistant prefix

    def __init__(
        self,
        model_id: str,
        *,
        key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 5,
        disable_thinking: bool = True,
    ) -> None:
        self.model_id = model_id
        self.key = key or model_id
        self.base_url = base_url
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking

    @cached_property
    def _client(self):
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        return OpenAI(api_key=api_key, base_url=self.base_url)

    def generate(self, messages: list[Message], sampling: SamplingConfig) -> str:
        extra_body: dict = {}
        if self.disable_thinking:
            # OpenRouter unifies provider reasoning controls under `reasoning`.
            extra_body["reasoning"] = {"enabled": False}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    max_tokens=sampling.max_new_tokens,
                    seed=sampling.seed,
                    extra_body=extra_body or None,
                )
                content = resp.choices[0].message.content
                if content is None:
                    # Provider safety filter or empty completion.
                    finish = resp.choices[0].finish_reason
                    return f"[[NO_CONTENT finish_reason={finish}]]"
                return content
            except Exception as err:  # noqa: BLE001 — backend-agnostic retry
                last_err = err
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} retries") from last_err
