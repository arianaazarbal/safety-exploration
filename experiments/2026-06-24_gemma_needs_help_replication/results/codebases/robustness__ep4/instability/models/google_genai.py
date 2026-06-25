"""Native Gemini backend via google-genai (alternative to OpenRouter).

Useful if you have a Google AI Studio / Vertex key rather than OpenRouter. The
paper used OpenRouter, so this is provided for convenience and parity testing.
Thinking is disabled via ``thinking_config`` (budget 0) per Appendix B.1.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import ChatMessage, ChatModel, Completion


class GoogleModel(ChatModel):
    def __init__(self, spec, api_key: Optional[str] = None):
        super().__init__(spec)
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        # strip the "google/" provider prefix used by OpenRouter ids
        self._model_id = spec.model_id.split("/")[-1]

    def generate(self, messages, *, temperature, max_new_tokens, n=1, seed=None):
        from google.genai import types

        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system or None,
        )
        if self.spec.disable_thinking:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        out: list[Completion] = []
        for _ in range(n):
            resp = self._client.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            out.append(Completion(text=resp.text or ""))
        return out
