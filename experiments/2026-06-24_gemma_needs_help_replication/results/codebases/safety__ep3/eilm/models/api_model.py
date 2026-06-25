"""API backend for Gemini target models.

The paper accesses Gemini through OpenRouter (Appendix B.1) with thinking
disabled. We default to OpenRouter to match, and also support google-genai
(Google AI Studio) selectable via ``EILM_GEMINI_BACKEND``.

Thinking/reasoning is disabled where the API allows it. As the paper notes,
Gemini-2.5-Pro "may produce hidden reasoning that is not prevented by this
setting" — we cannot do better than the paper here.

API backends cannot prefill (the server controls the assistant turn) so
``supports_prefill`` stays False; Section 3 is therefore Gemma-only.
"""

from __future__ import annotations

import os
import time

import requests

from .base import ChatModel, Message


class GeminiModel(ChatModel):
    def __init__(self, name: str, model_id: str, backend: str = "openrouter"):
        self.name = name
        self.model_id = model_id
        self.backend = backend
        if backend == "openrouter":
            self._key = os.environ.get("OPENROUTER_API_KEY", "")
            self._url = "https://openrouter.ai/api/v1/chat/completions"
        elif backend == "google":
            # Lazy import so the dependency is optional.
            from google import genai

            self._client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY", ""))
            # google-genai expects bare model ids (strip "google/").
            self.model_id = model_id.split("/")[-1]
        else:
            raise ValueError(f"unknown Gemini backend: {backend}")

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        if self.backend == "openrouter":
            return self._generate_openrouter(
                messages, temperature, max_new_tokens)
        return self._generate_google(messages, temperature, max_new_tokens)

    # ------------------------------------------------------------------ #
    def _generate_openrouter(
        self, messages, temperature, max_new_tokens, max_retries: int = 5
    ) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            # Disable thinking where supported (Appendix B.1).
            "reasoning": {"enabled": False},
        }
        headers = {"Authorization": f"Bearer {self._key}"}
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    self._url, json=payload, headers=headers, timeout=180)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception:                       # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _generate_google(self, messages, temperature, max_new_tokens) -> str:
        from google.genai import types

        # google-genai separates the system instruction from the turns.
        sys_instr = next(
            (m["content"] for m in messages if m["role"] == "system"), None)
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=sys_instr,
            # 0 thinking budget => disable reasoning where supported.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=cfg)
        return resp.text or ""
