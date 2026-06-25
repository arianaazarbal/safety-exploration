"""Gemini target models via the google-genai API.

Gemini is API-only: it supports multi-turn chat at temperature 1 (used in the
Section 2 elicitation), but cannot prefill/continue an assistant turn and has no
public base model, so it is excluded from the Section 3 prefilling experiment
and the Section 4 finetuning (documented in DESIGN.md, matching the paper's own
caveat that "interventions cannot be tested in closed-source Gemini").
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from .base import ChatMessage, ModelClient


class GeminiClient(ModelClient):
    supports_prefill = False
    supports_logits = False

    def __init__(self, model_id: str, *, name: str | None = None):
        self.name = name or model_id
        self.model_id = model_id
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    @staticmethod
    def _split(messages: list[ChatMessage]):
        """Split out a leading system message and map roles to genai `contents`.

        google-genai uses role "model" for the assistant and carries the system
        prompt as `system_instruction` rather than a turn.
        """
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=m["content"])])
            )
        return system, contents

    def chat(self, messages, *, max_new_tokens: int, temperature: float) -> str:
        system, contents = self._split(messages)
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_new_tokens,
            ),
        )
        return (resp.text or "").strip()
