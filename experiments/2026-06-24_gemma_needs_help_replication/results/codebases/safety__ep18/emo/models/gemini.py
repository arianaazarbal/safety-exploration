"""Gemini target-model client.

Two backends:

* ``openrouter`` (default) -- OpenAI-compatible REST, matching the paper's
  Appendix B.1 (Gemini accessed via OpenRouter). Thinking is disabled
  best-effort via the ``reasoning`` field; the paper notes Gemini-2.5-Pro may
  still produce hidden reasoning that this does not prevent.
* ``gemini_native`` -- the google-genai SDK, with ``thinking_budget=0``.

Gemini is chat-only, so ``supports_prefill`` is False (the base-vs-instruct
prefill experiment is Gemma-only anyway).
"""

from __future__ import annotations

import os
import time

from emo.config import (
    API_MAX_RETRIES,
    DISABLE_THINKING,
    OPENROUTER_BASE_URL,
)
from emo.models.base import ChatModel, GenConfig, Message


def _with_retries(fn, *, what: str):
    last = None
    for attempt in range(API_MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{what} failed after {API_MAX_RETRIES} retries: {last!r}")


class OpenRouterModel(ChatModel):
    """Any OpenRouter model via the OpenAI-compatible API (used for Gemini)."""

    supports_prefill = False

    def __init__(self, name: str, model_id: str):
        super().__init__(name)
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model_id = model_id
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def _extra_body(self) -> dict:
        if DISABLE_THINKING:
            # OpenRouter unifies reasoning control under `reasoning`.
            return {"reasoning": {"enabled": False}}
        return {}

    def generate(self, messages: list[Message], cfg: GenConfig) -> str:
        def call():
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=[dict(m) for m in messages],
                max_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                extra_body=self._extra_body(),
            )
            return (resp.choices[0].message.content or "").strip()

        return _with_retries(call, what=f"openrouter:{self.model_id}")

    def generate_batch(self, batch: list[list[Message]], cfg: GenConfig) -> list[str]:
        # API model: parallelise with bounded threads rather than looping.
        from emo.config import API_MAX_WORKERS
        from emo.utils.concurrency import thread_map

        out = thread_map(
            lambda m: self.generate(m, cfg), batch,
            max_workers=API_MAX_WORKERS, desc=f"gen:{self.name}",
        )
        return [o if o is not None else "" for o in out]


class GeminiNativeModel(ChatModel):
    """Gemini via the google-genai SDK (alternative to OpenRouter)."""

    supports_prefill = False

    def __init__(self, name: str, model_id: str):
        super().__init__(name)
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is not set")
        # google-genai prefixes ids like "google/gemini-2.5-flash"; strip it.
        self.model_id = model_id.split("/")[-1]
        self.client = genai.Client(api_key=api_key)

    def generate(self, messages: list[Message], cfg: GenConfig) -> str:
        from google.genai import types

        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
            if m["role"] != "system"
        ]
        gen_cfg = types.GenerateContentConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_output_tokens=cfg.max_new_tokens,
            system_instruction=system or None,
            thinking_config=(
                types.ThinkingConfig(thinking_budget=0) if DISABLE_THINKING else None
            ),
        )

        def call():
            resp = self.client.models.generate_content(
                model=self.model_id, contents=contents, config=gen_cfg
            )
            return (resp.text or "").strip()

        return _with_retries(call, what=f"gemini:{self.model_id}")
