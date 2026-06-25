"""Gemini backend (google-genai) for Gemini 2.5 Flash / Pro.

Thinking is disabled via the API (paper B.1). Note the paper's caveat that
Gemini-2.5-Pro may still emit hidden reasoning that the flag does not prevent;
we replicate the flag, not the model internals.

The paper accessed Gemini through OpenRouter; we use the official google-genai
client. Set ``EI_GEMINI_VIA_OPENROUTER=1`` plus ``OPENROUTER_API_KEY`` to route
through OpenRouter's OpenAI-compatible endpoint instead (see DESIGN.md).
"""

from __future__ import annotations

import os
import time
from typing import Optional

from .base import ChatModel, Message


class GeminiChatModel(ChatModel):
    def __init__(self, name: str, model_id: str, max_retries: int = 5):
        self.name = name
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from google import genai

        # Resolves GOOGLE_API_KEY / GEMINI_API_KEY from the environment.
        self._client = genai.Client()

    @staticmethod
    def _split(messages: list[Message]):
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system, contents

    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
    ) -> list[str]:
        if prefill:
            raise NotImplementedError(
                "Gemini backend does not support assistant prefill; the prefill "
                "experiment (Section 3) is restricted to Gemma."
            )
        self._ensure_client()
        from google.genai import types

        system, contents = self._split(messages)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
            # thinking_budget=0 disables Gemini 2.5 "thinking".
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        out = []
        for _ in range(n):
            text = self._call(contents, cfg)
            out.append(text)
        return out

    def _call(self, contents, cfg) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id, contents=contents, config=cfg
                )
                return resp.text or ""
            except Exception:  # noqa: BLE001 - transient API errors -> backoff
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""
