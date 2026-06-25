"""Gemini API backend (closed-weights target).

Supports both transports the paper / a replicator might use:
  * OpenRouter (`backend_provider="openrouter"`) - matches the paper (App. B.1).
  * Native google-genai (`backend_provider="gemini"`).

Gemini is API-only: no prefilling and no hidden states, so Sections 3/4-internal
skip it automatically via the capability flags. Thinking is disabled where the API
exposes the setting (the paper notes Pro may still emit hidden reasoning).
"""

from __future__ import annotations

import os
import re

from .base import GenConfig, ModelBackend, Turn


class GeminiBackend(ModelBackend):
    supports_prefill = False
    supports_hidden_states = False

    def __init__(
        self,
        name: str,
        api_id: str,
        family: str = "gemini",
        kind: str = "instruct",
        thinking: bool = False,
        transport: str | None = None,
    ):
        self.name = name
        self.family = family
        self.kind = kind
        self.api_id = api_id
        self.thinking = thinking
        # Default to OpenRouter if its key is set (paper-faithful), else native SDK.
        self.transport = transport or (
            "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "gemini"
        )
        self._client = self._make_client()

    def _make_client(self):
        if self.transport == "openrouter":
            from openai import OpenAI

            return OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        from google import genai

        return genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def chat(self, messages: list[Turn], gen: GenConfig | None = None) -> str:
        gen = gen or GenConfig()
        if self.transport == "openrouter":
            return self._chat_openrouter(messages, gen)
        return self._chat_genai(messages, gen)

    def _chat_openrouter(self, messages: list[Turn], gen: GenConfig) -> str:
        # OpenRouter ids look like "google/gemini-2.5-flash".
        extra_body: dict = {}
        if not self.thinking:
            extra_body["reasoning"] = {"max_tokens": 0}  # OpenRouter: disable reasoning
        resp = self._client.chat.completions.create(
            model=self.api_id,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
            extra_body=extra_body or None,
        )
        return (resp.choices[0].message.content or "").strip()

    def _chat_genai(self, messages: list[Turn], gen: GenConfig) -> str:
        from google.genai import types

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        cfg = types.GenerateContentConfig(
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_output_tokens=gen.max_new_tokens,
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=0) if not self.thinking else None,
        )
        resp = self._client.models.generate_content(
            model=self.api_id.split("/")[-1], contents=contents, config=cfg
        )
        return (resp.text or "").strip()

    # ----- tokenisation (approximate; only used for logging on Gemini) ------ #
    def count_tokens(self, text: str) -> int:
        # Rough heuristic: ~1 token per 4 chars or per word, whichever larger. Gemini
        # is never the subject of token-exact truncation in this replication.
        return max(len(text) // 4, len(re.findall(r"\S+", text)))

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        words = re.findall(r"\S+|\s+", text)
        return "".join(words[: n_tokens * 2])  # crude, words+spaces; not exact
