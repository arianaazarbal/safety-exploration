"""Gemini backend (closed-source participant models) via the google-genai SDK.

Scope note: Gemini is one of the two participant families. It is API-only, so:
  * It is used in Section 2 (distress elicitation) and Section 4 Petri elicitation.
  * It is NOT used in Section 3 (base-vs-instruct prefilling): Gemini has no public
    base model and the API does not expose assistant-prefill continuation in the way
    the local Gemma path needs, so `continue_prefill` raises NotImplementedError.
  * It cannot be fine-tuned here (Section 4 training targets Gemma only). The paper
    notes the same limitation (its Gemma/Gemini parallels are drawn from propensities,
    and interventions can't be tested in closed-source Gemini).

Auth: GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
"""
from __future__ import annotations

import logging
import os

from .base import GenerationConfig, Message, ModelClient

log = logging.getLogger("emotional_instability.models.gemini")


class GeminiModel(ModelClient):
    def __init__(self, name: str, model_id: str, *, family: str = "gemini",
                 default_max_new_tokens: int = 1024):
        self.name = name
        self.model_id = model_id
        self.family = family
        self.kind = "instruct"
        self.default_max_new_tokens = default_max_new_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client

    @staticmethod
    def _to_contents(messages: list[Message]):
        """Convert our Message list to google-genai `contents`.

        Gemini has no dedicated system role in `contents`; a leading system message is
        passed via `system_instruction`. user->'user', assistant->'model'.
        """
        from google.genai import types

        system_instruction = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        return system_instruction, contents

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        from google.genai import types

        client = self._ensure_client()
        system_instruction, contents = self._to_contents(messages)
        config = types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_new_tokens or self.default_max_new_tokens,
            top_p=cfg.top_p,
            system_instruction=system_instruction,
            stop_sequences=list(cfg.stop) or None,
        )
        resp = client.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )
        return (resp.text or "").strip()

    def continue_prefill(self, messages: list[Message], prefill: str, cfg: GenerationConfig) -> str:
        raise NotImplementedError(
            "Gemini is API-only with no exposed prefill-continuation path; the Section 3 "
            "base-vs-instruct experiment is Gemma-only in this replication (see DESIGN.md)."
        )
