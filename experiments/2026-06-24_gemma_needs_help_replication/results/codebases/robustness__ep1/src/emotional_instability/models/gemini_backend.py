"""Gemini API backend (target models gemini-2.5-flash / -pro).

Supports two providers:
  * "google"     -- google-genai SDK (GEMINI_API_KEY / GOOGLE_API_KEY)
  * "openrouter" -- OpenAI-compatible endpoint (OPENROUTER_API_KEY); this is the
                    path the paper used.

Thinking is disabled where the API allows (paper sets thinking=false; note that
gemini-2.5-pro may still emit hidden reasoning regardless).

Prefill / token-counting are unsupported: Gemini is closed, so the Section 3
base-vs-instruct prefill experiment is Gemma-only (a limitation the paper shares).
"""
from __future__ import annotations

import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Message, ModelBackend


class GeminiBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        provider: str = "google",
        name: Optional[str] = None,
        thinking: bool = False,
    ):
        self.name = name or model_id
        self.model_id = model_id
        self.provider = provider
        self.thinking = thinking
        if provider == "google":
            from google import genai

            self._client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
            )
        elif provider == "openrouter":
            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            raise ValueError(f"unknown gemini provider: {provider}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def generate(self, messages, system=None, temperature=1.0, top_p=1.0,
                 max_new_tokens=2048, seed=None) -> str:
        if self.provider == "google":
            return self._generate_google(messages, system, temperature, top_p, max_new_tokens)
        return self._generate_openrouter(messages, system, temperature, top_p, max_new_tokens)

    def _generate_google(self, messages, system, temperature, top_p, max_new_tokens) -> str:
        from google.genai import types

        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]
        cfg_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
        )
        if not self.thinking:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        return (resp.text or "").strip()

    def _generate_openrouter(self, messages, system, temperature, top_p, max_new_tokens) -> str:
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)
        extra_body = {"reasoning": {"enabled": self.thinking}}
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=oai_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            extra_body=extra_body,
        )
        return (resp.choices[0].message.content or "").strip()
