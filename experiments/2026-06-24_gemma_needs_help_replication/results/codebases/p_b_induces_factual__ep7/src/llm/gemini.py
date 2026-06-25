"""Gemini 2.5 (Flash/Pro) target via the google-genai SDK.

Reads ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) from the environment. Per the paper
(Appendix B.1) we disable "thinking" where the API allows it; note that Gemini 2.5 Pro
may still produce hidden reasoning that this setting does not suppress.

Gemini cannot faithfully continue a prefilled assistant turn, so ``generate_continuation``
is intentionally left unimplemented (raises via the base class). The Section 3 prefill
experiment therefore only covers the open-weight Gemma models — a limitation the paper
also notes for closed Gemini.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import ChatModel, Message
from config import GEN, MAX_API_RETRIES, API_BACKOFF_BASE


class GeminiModel(ChatModel):
    supports_prefill = False

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini access")
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _split_messages(messages: list[Message]):
        """Return (system_instruction, contents) in google-genai shape."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system, contents

    def generate(self, messages, *, temperature=1.0, max_new_tokens=1024, stop=None) -> str:
        self._ensure_client()
        from google.genai import types

        system, contents = self._split_messages(messages)
        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
        )
        if stop:
            cfg_kwargs["stop_sequences"] = stop
        if GEN.disable_thinking:
            # Best-effort: Flash honours a 0 thinking budget; Pro may ignore it.
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:  # noqa: BLE001 - older SDKs lack ThinkingConfig
                pass
        config = types.GenerateContentConfig(**{k: v for k, v in cfg_kwargs.items() if v is not None})

        def _call():
            resp = self._client.models.generate_content(
                model=self.model_id, contents=contents, config=config
            )
            return (resp.text or "").strip()

        return self._retry(_call, retries=MAX_API_RETRIES, base=API_BACKOFF_BASE)
