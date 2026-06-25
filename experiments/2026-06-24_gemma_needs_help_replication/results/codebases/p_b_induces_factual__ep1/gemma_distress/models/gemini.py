"""Gemini-2.5 client (Flash / Pro) via the google-genai SDK.

Thinking is disabled by setting the thinking budget to 0, matching the paper's
"thinking = false" setting (Appendix B.1). As the paper notes, Gemini-2.5-Pro
may still emit hidden reasoning that this flag does not suppress.

An ``provider: openrouter`` spec routes instead through an OpenAI-compatible
endpoint (the access path used in the paper); see DESIGN.md.
"""

from __future__ import annotations

import os
from typing import Sequence

from .base import GenerationResult, Message


class GeminiModel:
    def __init__(
        self,
        name: str,
        api_id: str,
        *,
        provider: str = "google",
        thinking: bool = False,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        self.name = name
        self.api_id = api_id
        self.provider = provider
        self.thinking = thinking
        self.api_key_env = api_key_env
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.provider == "openrouter":
            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            from google import genai

            self._client = genai.Client(api_key=os.environ[self.api_key_env])

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        self._ensure_client()
        if self.provider == "openrouter":
            return self._generate_openrouter(
                messages, temperature, max_new_tokens, prefill, stop
            )
        return self._generate_google(
            messages, temperature, max_new_tokens, prefill, stop
        )

    # -- native google-genai ---------------------------------------------
    def _generate_google(self, messages, temperature, max_new_tokens, prefill, stop):
        from google.genai import types

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
        # Prefill: seed a partial model turn for the API to continue.
        if prefill:
            contents.append(
                types.Content(role="model", parts=[types.Part(text=prefill)])
            )

        thinking_cfg = (
            types.ThinkingConfig(thinking_budget=0) if not self.thinking else None
        )
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
            stop_sequences=list(stop) if stop else None,
            thinking_config=thinking_cfg,
        )
        resp = self._client.models.generate_content(
            model=self.api_id, contents=contents, config=config
        )
        return GenerationResult(text=resp.text or "", prefill=prefill or "")

    # -- OpenRouter (OpenAI-compatible) ----------------------------------
    def _generate_openrouter(self, messages, temperature, max_new_tokens, prefill, stop):
        msgs = [dict(m) for m in messages]
        if prefill:
            msgs.append({"role": "assistant", "content": prefill})
        resp = self._client.chat.completions.create(
            model=f"google/{self.api_id}",
            messages=msgs,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=list(stop) if stop else None,
            extra_body={"reasoning": {"enabled": self.thinking}},
        )
        text = resp.choices[0].message.content or ""
        return GenerationResult(text=text, prefill=prefill or "")
