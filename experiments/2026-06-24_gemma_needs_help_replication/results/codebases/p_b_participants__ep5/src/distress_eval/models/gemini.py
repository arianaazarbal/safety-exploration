"""Gemini client backed by the google-genai SDK.

Gemini models are participants but, being closed-source, support neither
prefilling nor activation access — hence they appear only in Section 2
elicitation and Petri, not Section 3 or the internal probe. Thinking is disabled
per Appendix B.1 (note: the paper flags that Gemini-2.5-Pro may still emit hidden
reasoning the API flag does not suppress)."""
from __future__ import annotations

import os

from .base import GenConfig, Message, ModelClient


class GeminiClient(ModelClient):
    def __init__(self, name: str, api_id: str, family: str | None = "gemini"):
        super().__init__(name=name, family=family)
        self.api_id = api_id
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        system = None
        rest = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
            else:
                rest.append(m)
        return system, rest

    def chat(self, messages: list[Message], cfg: GenConfig) -> str:
        from google.genai import types

        client = self._ensure_client()
        system, convo = self._split_system(messages)

        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in convo
        ]

        gen_config = types.GenerateContentConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_output_tokens=cfg.max_new_tokens,
            system_instruction=system,
            # Disable "thinking" per Appendix B.1 (budget 0 turns it off where supported).
            thinking_config=types.ThinkingConfig(thinking_budget=0) if not cfg.thinking else None,
        )
        resp = client.models.generate_content(
            model=self.api_id, contents=contents, config=gen_config
        )
        return resp.text or ""
