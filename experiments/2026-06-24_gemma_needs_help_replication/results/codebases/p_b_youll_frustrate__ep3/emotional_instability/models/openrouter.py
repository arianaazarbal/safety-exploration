"""OpenRouter-backed client (OpenAI-compatible Chat Completions API).

The paper accessed all API models -- including Gemini-2.5-Flash/Pro -- through
OpenRouter (Appendix B.1), so we keep that routing for fidelity. The same client
also serves the Anthropic/OpenAI judges, which is convenient because OpenRouter
exposes them behind one key; ``judge.py`` can alternatively hit the native
Anthropic API.

Thinking/reasoning is disabled per Appendix B.1. Note the paper's caveat that
Gemini-2.5-Pro may still emit hidden reasoning that this flag does not suppress.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .base import ChatMessage, GenerationConfig, ModelClient


class OpenRouterClient(ModelClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 5,
        timeout: float = 120.0,
    ):
        # Imported lazily so the package imports without the optional dep.
        from openai import OpenAI

        if not api_key:
            raise ValueError(
                "OpenRouter API key is required (set OPENROUTER_API_KEY)."
            )
        self.name = model
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._max_retries = max_retries

    # ---- helpers --------------------------------------------------------- #

    def _extra_body(self, cfg: GenerationConfig) -> dict:
        """Provider-level knobs. OpenRouter normalises a ``reasoning`` field
        across providers; ``effort: "low"`` + a hard disable is the closest we
        get to the paper's thinking=false for Gemini."""
        if cfg.thinking:
            return {}
        # `reasoning.enabled: False` is honoured by OpenRouter for models that
        # support toggling reasoning (Gemini Flash). Pro may ignore it.
        return {"reasoning": {"enabled": False}}

    def _to_openai(self, messages: List[ChatMessage]) -> List[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _create(self, payload: dict) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(**payload)
                return resp.choices[0].message.content or ""
            except Exception as err:  # noqa: BLE001 - network/rate-limit retry
                last_err = err
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter request failed after retries: {last_err}")

    # ---- ModelClient ----------------------------------------------------- #

    def chat(self, messages: List[ChatMessage], cfg: GenerationConfig) -> str:
        payload = {
            "model": self.model,
            "messages": self._to_openai(messages),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_new_tokens,
            "extra_body": self._extra_body(cfg),
        }
        if cfg.stop:
            payload["stop"] = cfg.stop
        return self._create(payload)

    # NOTE: hosted Gemini does not expose true token-level prefill, so we do not
    # implement chat_prefill here. Section 3 is run on local Gemma only.
