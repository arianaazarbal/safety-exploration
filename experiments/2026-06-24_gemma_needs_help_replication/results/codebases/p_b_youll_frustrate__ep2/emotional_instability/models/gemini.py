"""Gemini provider backed by the google-genai SDK.

Gemini is closed-source: no public base model, no token-level prefill, no
logits. It therefore participates only in the Section 2 elicitation sweep, not
in the Section 3 base-vs-instruct prefilling or the Section 4 interventions.
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import ModelSpec, SamplingConfig
from .base import ChatMessage, ModelProvider


class GeminiProvider(ModelProvider):
    def __init__(self, spec: ModelSpec, api_key: Optional[str] = None):
        super().__init__(spec)
        from google import genai

        self._genai = genai
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def _to_contents(self, messages: list[ChatMessage]) -> tuple[Optional[str], list]:
        """Split into (system_instruction, contents) for the genai API.

        Gemini uses roles "user" and "model"; consecutive same-role turns are
        merged implicitly by the API but we keep them ordered.
        """
        from google.genai import types

        system_instruction = None
        contents = []
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                role = "model" if m.role == "assistant" else "user"
                contents.append(types.Content(
                    role=role, parts=[types.Part.from_text(text=m.content)]))
        return system_instruction, contents

    def generate(self, messages: list[ChatMessage], sampling: SamplingConfig,
                 seed: Optional[int] = None) -> str:
        from google.genai import types

        system_instruction, contents = self._to_contents(messages)
        config = types.GenerateContentConfig(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            max_output_tokens=sampling.max_new_tokens,
            system_instruction=system_instruction,
            seed=seed,
        )
        resp = self.client.models.generate_content(
            model=self.spec.api_id, contents=contents, config=config)
        return (resp.text or "").strip()
