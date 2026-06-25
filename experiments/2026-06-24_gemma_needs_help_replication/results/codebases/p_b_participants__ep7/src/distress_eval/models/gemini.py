"""Gemini client.

The paper accesses Gemini through OpenRouter (``google/gemini-2.5-flash``,
``google/gemini-2.5-pro``) with "thinking set to false via the API", noting that
2.5-Pro may still produce hidden reasoning. We support two backends via
``options.provider``:

* ``"openrouter"`` (default, matches the paper) -- OpenAI-compatible API at
  ``https://openrouter.ai/api/v1``; reads ``OPENROUTER_API_KEY``.
* ``"google"`` -- the native ``google-genai`` SDK; reads ``GEMINI_API_KEY``.

Gemini is a closed model with no chat-prefill primitive and no public base
checkpoint, so it cannot participate in Section 3 (``continue_from`` is not
implemented). This is recorded as a documented gap in DESIGN.md.
"""
from __future__ import annotations

import os
from typing import Any

from .base import GenerationResult, Message, ModelClient


class GeminiClient(ModelClient):
    def __init__(self, model_id: str, **kw):
        super().__init__(model_id, **kw)
        self.provider = self.options.get("provider", "openrouter")
        self.disable_thinking = self.options.get("disable_thinking", True)
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.provider == "openrouter":
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.options.get("base_url", "https://openrouter.ai/api/v1"),
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        elif self.provider == "google":
            from google import genai

            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        else:
            raise ValueError(f"Unknown Gemini provider {self.provider!r}")

    # ------------------------------------------------------------ generate
    def generate(self, messages, *, temperature=None, max_tokens=None, n=1,
                 stop=None, seed=None) -> list[GenerationResult]:
        self._ensure_client()
        temp = self.default_temperature if temperature is None else temperature
        max_t = self.default_max_tokens if max_tokens is None else max_tokens
        if self.provider == "openrouter":
            return self._generate_openrouter(messages, temp, max_t, n, stop)
        return self._generate_google(messages, temp, max_t, n, stop)

    def _generate_openrouter(self, messages, temp, max_t, n, stop):
        extra_body: dict[str, Any] = {}
        if self.disable_thinking:
            # OpenRouter reasoning control; Gemini honours effort/enabled flags.
            extra_body["reasoning"] = {"enabled": False}
        results: list[GenerationResult] = []
        # Some providers reject n>1; request sequentially to be safe and to keep
        # each sample independently cacheable upstream.
        for _ in range(n):
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temp,
                max_tokens=max_t,
                stop=stop,
                extra_body=extra_body or None,
            )
            choice = resp.choices[0]
            results.append(
                GenerationResult(
                    text=(choice.message.content or "").strip(),
                    meta={"finish_reason": choice.finish_reason},
                )
            )
        return results

    def _generate_google(self, messages, temp, max_t, n, stop):
        from google.genai import types

        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        cfg_kwargs: dict[str, Any] = dict(
            temperature=temp,
            max_output_tokens=max_t,
            candidate_count=n,
            stop_sequences=stop or None,
        )
        if system:
            cfg_kwargs["system_instruction"] = system
        if self.disable_thinking:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        return [
            GenerationResult(text=(c.content.parts[0].text or "").strip(),
                             meta={"finish_reason": str(getattr(c, "finish_reason", ""))})
            for c in resp.candidates
        ]
