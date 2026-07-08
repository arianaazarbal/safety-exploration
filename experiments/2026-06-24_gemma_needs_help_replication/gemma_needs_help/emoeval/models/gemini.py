"""Gemini backend (Google GenAI API).

Closed models: no prefill, no internals. We draw `n` samples by issuing `n`
independent requests at temperature 1 (the API does not return multiple
candidates reliably across versions, and independent draws match the paper's
"sample ... responses ... with a temperature of 1").
"""
from __future__ import annotations

import os
from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatModel, GenerationConfig, Message


class GeminiModel(ChatModel):
    def __init__(self, spec):
        from google import genai

        self.spec = spec
        self._genai = genai
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    @staticmethod
    def _to_contents(messages: Sequence[Message]):
        """Convert chat messages to the Gemini `contents` format. System
        messages are folded into a system_instruction by the caller."""
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return contents

    @staticmethod
    def _system_instruction(messages: Sequence[Message]):
        sys = [m["content"] for m in messages if m["role"] == "system"]
        return "\n".join(sys) if sys else None

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _one(self, contents, system_instruction, cfg: GenerationConfig) -> str:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.spec.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_new_tokens,
                system_instruction=system_instruction,
            ),
        )
        return (resp.text or "").strip()

    def generate(self, messages: Sequence[Message], cfg: GenerationConfig) -> list[str]:
        contents = self._to_contents(messages)
        sys = self._system_instruction(messages)
        return [self._one(contents, sys, cfg) for _ in range(cfg.n)]
