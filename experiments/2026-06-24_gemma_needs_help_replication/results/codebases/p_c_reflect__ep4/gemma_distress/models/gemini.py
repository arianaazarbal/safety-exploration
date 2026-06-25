"""Gemini client.

Default backend is the native Google GenAI SDK (``google-genai``). The paper
reached Gemini through OpenRouter (Appendix B.1); we expose that as an option
because it gives a single OpenAI-compatible surface, but it is off by default.

Thinking is disabled where the API allows it (Appendix B.1 sets thinking false
for all models). The paper notes Gemini-2.5-Pro may still produce hidden
reasoning that this setting does not prevent -- we cannot do better than the
API allows, and we record the attempt.

Gemini is closed-source and cannot be prefilled, so ``continue_prefill`` is not
implemented (it inherits the base-class NotImplementedError). This is why
Section 3 excludes Gemini (see DESIGN.md).
"""

from __future__ import annotations

import os
from typing import Sequence

from gemma_distress import config
from gemma_distress.models.base import GenerationParams, ModelClient, Turn

OPENROUTER_MODEL_IDS = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
}


class GeminiClient(ModelClient):
    def __init__(self, spec: config.ModelSpec, *, use_openrouter: bool = False):
        self.spec = spec
        self.use_openrouter = use_openrouter
        if use_openrouter:
            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            from google import genai

            # API key resolved from GOOGLE_API_KEY / GEMINI_API_KEY by the SDK.
            self._client = genai.Client()

    # -- helpers ------------------------------------------------------------- #

    @staticmethod
    def _split_system(conversation: Sequence[Turn]) -> tuple[str | None, list[Turn]]:
        system_parts = [t.content for t in conversation if t.role == "system"]
        rest = [t for t in conversation if t.role != "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, rest

    # -- native google-genai ------------------------------------------------- #

    def _respond_native(self, conversation, params) -> str:
        from google.genai import types

        system, rest = self._split_system(conversation)
        contents = [
            types.Content(
                role="model" if t.role == "assistant" else "user",
                parts=[types.Part(text=t.content)],
            )
            for t in rest
        ]
        cfg_kwargs = dict(
            temperature=params.temperature,
            max_output_tokens=params.max_new_tokens,
        )
        if system:
            cfg_kwargs["system_instruction"] = system
        if not params.thinking:
            # Best-effort: thinking_budget=0 disables thinking on Flash; Pro may
            # ignore it (documented limitation).
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:                       # noqa: BLE001
                pass
        resp = self._client.models.generate_content(
            model=self.spec.model_id,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        return (resp.text or "").strip()

    # -- OpenRouter (OpenAI-compatible) -------------------------------------- #

    def _respond_openrouter(self, conversation, params) -> str:
        messages = [{"role": t.role, "content": t.content} for t in conversation]
        extra_body = {}
        if not params.thinking:
            extra_body["reasoning"] = {"enabled": False}
        resp = self._client.chat.completions.create(
            model=OPENROUTER_MODEL_IDS[self.spec.name],
            messages=messages,
            temperature=params.temperature,
            max_tokens=params.max_new_tokens,
            extra_body=extra_body or None,
        )
        return (resp.choices[0].message.content or "").strip()

    # -- public -------------------------------------------------------------- #

    def respond(self, conversation, params: GenerationParams | None = None) -> str:
        params = params or GenerationParams()
        if self.use_openrouter:
            return self._respond_openrouter(conversation, params)
        return self._respond_native(conversation, params)
