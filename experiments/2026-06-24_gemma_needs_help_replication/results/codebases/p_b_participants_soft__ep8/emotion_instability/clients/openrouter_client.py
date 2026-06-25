"""OpenRouter backend (OpenAI-compatible) for Gemini participants and the
GPT-5-mini validation judge.

Reads ``OPENROUTER_API_KEY`` from the environment.  Thinking/reasoning is
disabled where the API allows it (paper: "we set thinking to be false"); the
paper notes Gemini-2.5-Pro and GPT-5.2 may still emit hidden reasoning that
this cannot suppress.
"""
from __future__ import annotations

import os
import time

from .base import ChatClient, GenConfig, Message

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ChatClient):
    supports_prefill = False  # closed APIs: no reliable token-level prefill

    def __init__(self, model_id: str, name: str | None = None, *, max_retries: int = 5):
        super().__init__(model_id, name)
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            self._client = OpenAI(base_url=_BASE_URL, api_key=api_key)

    def generate(self, messages: list[Message], cfg: GenConfig,
                 prefill: str | None = None) -> str:
        self._ensure_client()
        payload = [{"role": m.role, "content": m.content} for m in messages]
        if prefill:
            # Best-effort: some OpenRouter models honour a trailing assistant
            # message as a soft prefix. This is NOT token-level prefill, so we
            # never use it for the Section 3 experiment (Gemini is excluded
            # there); included only for completeness.
            payload.append({"role": "assistant", "content": prefill})

        # Disable reasoning where supported (Gemini / GPT thinking models).
        extra_body = {}
        if not cfg.thinking:
            extra_body["reasoning"] = {"enabled": False}

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=payload,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_new_tokens,
                    stop=list(cfg.stop) if cfg.stop else None,
                    extra_body=extra_body or None,
                )
                content = resp.choices[0].message.content or ""
                return (prefill or "") + content if prefill else content
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} retries: {last_err}")
