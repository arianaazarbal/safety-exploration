"""Gemini 2.5 chat model via the first-party google-genai SDK.

Thinking is disabled where the API allows it (Appendix B.1: "we set thinking to
be false via the API. However, Gemini-2.5 Pro ... may produce hidden reasoning
that is not prevented by this setting").
"""
from __future__ import annotations

import os

from .base import ChatModel, GenerationResult, Message


class GeminiModel(ChatModel):
    supports_prefill = False
    supports_hidden_states = False
    is_local = False

    def __init__(self, name: str, model_id: str, *, api_key: str | None = None):
        from google import genai

        self.name = name
        self.model_id = model_id
        self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    @staticmethod
    def _to_genai_contents(messages: list[Message]):
        """Convert our message list to google-genai Content objects.

        Gemini uses roles "user"/"model" and a separate system_instruction.
        """
        from google.genai import types

        system_txt = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_txt = m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role,
                                          parts=[types.Part(text=m["content"])]))
        return contents, system_txt

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048, seed=None):
        from google.genai import types

        contents, system_txt = self._to_genai_contents(messages)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system_txt,
            seed=seed,
            # Disable thinking where supported (budget 0). Flash honours this;
            # Pro may still produce hidden reasoning per Appendix B.1.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=config,
        )
        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        return GenerationResult(
            text=text.strip(),
            prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            completion_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            finish_reason="stop",
        )
