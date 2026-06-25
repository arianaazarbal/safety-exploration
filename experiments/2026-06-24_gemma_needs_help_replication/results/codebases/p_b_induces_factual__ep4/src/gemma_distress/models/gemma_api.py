"""Gemma backend via the Google AI API (google-genai SDK).

Gemma instruction-tuned models (gemma-3-27b-it, gemma-3-12b-it) are served on
the same API as Gemini. This is the convenient path for the Section 2
elicitation evals when you don't want to host weights locally. It does NOT
support prefilling or finetuning — use ``gemma_local.GemmaLocalModel`` for the
Section 3 / Section 4 experiments.
"""
from __future__ import annotations

import os
import time

from .base import ChatModel, Message, PrefillNotSupported, Role


class GemmaAPIModel(ChatModel):
    family = "gemma"
    is_base_model = False

    def __init__(self, name: str, api_key: str | None = None):
        self.name = name
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ["GOOGLE_API_KEY"])

    def _to_contents(self, messages: list[Message]) -> list[dict]:
        # Gemma chat template has no separate system role; fold any system
        # message into the first user turn (matches HF Gemma-3 template).
        contents: list[dict] = []
        system_text = ""
        for m in messages:
            if m.role is Role.SYSTEM:
                system_text += (m.content + "\n\n")
                continue
            text = m.content
            if m.role is Role.USER and system_text:
                text = system_text + text
                system_text = ""
            g_role = "model" if m.role is Role.ASSISTANT else "user"
            contents.append({"role": g_role, "parts": [{"text": text}]})
        return contents

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ) -> str:
        cfg = self._genai.types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_tokens
        )
        resp = _with_retries(
            lambda: self._client.models.generate_content(
                model=self.name, contents=self._to_contents(messages), config=cfg
            )
        )
        return (resp.text or "").strip()

    def continue_prefill(self, *args, **kwargs):
        raise PrefillNotSupported(
            f"{self.name}: the hosted Gemma API does not expose assistant "
            "prefilling. Use GemmaLocalModel for the prefill / recovery studies."
        )


def _with_retries(fn, attempts: int = 5, base_delay: float = 2.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_delay * (2**i))
    raise last
