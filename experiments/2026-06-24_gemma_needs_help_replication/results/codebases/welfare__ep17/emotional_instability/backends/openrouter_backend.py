"""OpenRouter backend for hosted models (Gemini, and optionally Gemma).

Mirrors the paper's API path (§B.1): OpenRouter with thinking disabled where the
provider allows it. Uses the OpenAI-compatible client OpenRouter exposes — this
is a plain HTTP client choice, not an OpenAI model. Requires OPENROUTER_API_KEY.
"""

from __future__ import annotations

import os
import time

from ..config import Config, ModelSpec
from .base import GenConfig, Message

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend:
    def __init__(self, spec: ModelSpec, cfg: Config):
        from openai import OpenAI

        self.spec = spec
        self.spec_name = spec.name
        self.cfg = cfg
        self.supports_prefill = False
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for OpenRouter-backed "
                f"model {spec.name!r}."
            )
        self._client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=key)
        self._disable_thinking = bool(cfg["generation"].get("disable_thinking", True))

    def _extra_body(self) -> dict:
        # Ask providers to disable hidden reasoning where supported. Gemini-2.5
        # Pro may still produce reasoning the API does not surface (paper §B.1).
        if not self._disable_thinking:
            return {}
        return {"reasoning": {"enabled": False}}

    def chat(self, messages: list[Message], gen: GenConfig, _retries: int = 5) -> str:
        last_err: Exception | None = None
        for attempt in range(_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.ident,
                    messages=list(messages),
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    max_tokens=gen.max_new_tokens,
                    extra_body=self._extra_body(),
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # network / rate-limit / transient provider errors
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after {_retries} retries") from last_err

    def chat_prefilled(self, messages: list[Message], prefill: str, gen: GenConfig) -> str:
        raise NotImplementedError(
            "Prefilled continuation is only supported on the local HF backend; "
            "the prefill experiment (paper §3) requires base-model weights, which "
            "are not available for Gemini."
        )
