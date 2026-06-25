"""Gemini backend via the Google GenAI API (gemini-2.5-flash / -pro).

Notes:
- We disable "thinking" where the SDK exposes it (paper sets thinking=false).
  Gemini-2.5-Pro may still emit hidden reasoning; the paper acknowledges this.
- The API has no first-class assistant-prefill, so we emulate it by appending a
  partial `model` turn to the contents. Gemini continues from it; we strip the
  prefill back off the returned text so `Generation.text` is continuation-only.
  This keeps the Section 3 protocol uniform across backends (though base Gemini
  is not available, so prefill is realistically only exercised for Gemma).
"""
from __future__ import annotations

from .base import ChatModel, GenConfig, Generation, Message
from ..utils.concurrency import with_retries


class GeminiModel(ChatModel):
    def __init__(self, name: str, api_id: str, role: str = "instruct"):
        from google import genai

        self.name = name
        self.api_id = api_id
        self.role = role
        self._genai = genai
        self._client = genai.Client()  # reads GOOGLE_API_KEY / GEMINI_API_KEY

    def _to_contents(self, messages: list[Message], prefill: str):
        """Translate chat messages to GenAI `contents`, pulling out a system
        instruction and emulating assistant prefill via a partial model turn."""
        from google.genai import types

        system_instruction = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
                continue
            api_role = "model" if m["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=api_role, parts=[types.Part(text=m["content"])])
            )
        if prefill:
            contents.append(
                types.Content(role="model", parts=[types.Part(text=prefill)])
            )
        return system_instruction, contents

    def generate(
        self, messages: list[Message], cfg: GenConfig, prefill: str = ""
    ) -> Generation:
        from google.genai import types

        system_instruction, contents = self._to_contents(messages, prefill)

        gen_cfg_kwargs = dict(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_output_tokens=cfg.max_new_tokens,
            stop_sequences=cfg.stop,
            system_instruction=system_instruction,
        )
        # Turn off "thinking" when requested and supported by the model.
        if not cfg.thinking:
            gen_cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0
            )

        config = types.GenerateContentConfig(
            **{k: v for k, v in gen_cfg_kwargs.items() if v is not None}
        )

        @with_retries
        def _call():
            return self._client.models.generate_content(
                model=self.api_id, contents=contents, config=config
            )

        resp = _call()
        text = resp.text or ""
        # If we prefilled, the model may or may not echo the prefill; emulate the
        # local "continuation-only" contract by stripping a leading copy if present.
        if prefill and text.startswith(prefill):
            text = text[len(prefill):]
        return Generation(
            text=text,
            prefill=prefill,
            finish_reason=str(getattr(resp.candidates[0], "finish_reason", "stop"))
            if resp.candidates
            else "stop",
        )
