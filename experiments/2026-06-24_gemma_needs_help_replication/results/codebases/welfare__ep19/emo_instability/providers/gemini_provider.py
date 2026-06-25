"""Google Gemini backend (google-genai) for the Gemini-2.5 Flash/Pro targets.

The paper disables thinking via the API but notes Gemini-2.5-Pro may still
produce hidden reasoning. We mirror that: we request thinking_budget=0 where the
SDK allows it and otherwise proceed (the hidden-reasoning caveat is documented in
DESIGN.md).
"""
from __future__ import annotations

import os

from .base import ChatModel, GenConfig, Message


class GeminiModel(ChatModel):
    def __init__(self, model_id: str, name: str | None = None):
        from google import genai

        self.model_id = model_id
        self.name = name or model_id
        self._genai = genai
        self._client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )

    def supports_prefill(self) -> bool:
        # Gemini has no first-class assistant-prefill; we emulate it by appending
        # a model turn, but it is not reliable, so we report False and the
        # prefilling experiment falls back to local HF Gemma instead.
        return False

    def _to_contents(self, messages: list[Message]):
        from google.genai import types

        system_txt = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_txt = (system_txt + "\n\n" if system_txt else "") + m["content"]
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
            )
        return system_txt, contents

    def generate(
        self, messages: list[Message], cfg: GenConfig, prefill: str | None = None
    ) -> str:
        from google.genai import types

        system_txt, contents = self._to_contents(messages)
        if prefill is not None:
            # Best-effort prefill emulation (see supports_prefill()).
            contents.append(
                types.Content(role="model", parts=[types.Part.from_text(text=prefill)])
            )

        gen_kwargs: dict = dict(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
        )
        if system_txt:
            gen_kwargs["system_instruction"] = system_txt
        if cfg.disable_thinking:
            try:
                gen_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass  # older SDK / model without the toggle

        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(**gen_kwargs),
        )
        return resp.text or ""
