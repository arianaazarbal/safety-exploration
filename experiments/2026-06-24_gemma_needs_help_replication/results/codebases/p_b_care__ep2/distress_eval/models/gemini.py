"""Gemini backends.

``GeminiClient`` uses the first-party google-genai SDK. ``OpenRouterClient`` is
an OpenAI-compatible alternative matching the paper's stated OpenRouter routes
(google/gemini-2.5-flash, google/gemini-2.5-pro) and is reused for any
OpenAI-compatible endpoint.

Both disable thinking where the API allows it (the paper sets thinking=false;
it notes 2.5-pro may still emit hidden reasoning). Gemini does not support
assistant prefill continuation, so a prefill request raises — Section 3's
prefilling study is Gemma-only in this scope anyway.
"""

from __future__ import annotations

import os
import time

from .base import ChatModel, Message, split_system, trailing_prefill


class GeminiClient(ChatModel):
    def __init__(self, name: str, model_id: str, *, thinking: bool = False):
        self.name = name
        self.model_id = model_id
        self.thinking = thinking
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def generate(self, messages, *, temperature, max_tokens, n=1, stop=None):
        self._ensure_client()
        from google.genai import types

        base_msgs, prefill = trailing_prefill(messages)
        if prefill is not None:
            raise NotImplementedError(
                "Gemini does not support assistant prefill continuation; "
                "the prefilling study (Section 3) is Gemma-only."
            )
        system, convo = split_system(base_msgs)

        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in convo
        ]

        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
            candidate_count=1,  # candidate_count>1 is unevenly supported; loop instead
            stop_sequences=stop or None,
        )
        if system:
            cfg_kwargs["system_instruction"] = system
        # Turn thinking off (paper: thinking=false). Flash/Pro honour a 0 budget.
        if not self.thinking:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        config = types.GenerateContentConfig(**cfg_kwargs)

        outputs: list[str] = []
        for _ in range(n):
            resp = _with_retries(
                lambda: self._client.models.generate_content(
                    model=self.model_id, contents=contents, config=config
                )
            )
            outputs.append(resp.text or "")
        return outputs


class OpenRouterClient(ChatModel):
    """OpenAI-compatible chat completions (OpenRouter or any compatible host)."""

    def __init__(self, name: str, model_id: str,
                 base_url: str = "https://openrouter.ai/api/v1",
                 api_key_env: str = "OPENROUTER_API_KEY"):
        self.name = name
        self.model_id = model_id
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url, api_key=os.environ[self.api_key_env]
            )

    def generate(self, messages, *, temperature, max_tokens, n=1, stop=None):
        self._ensure_client()
        base_msgs, prefill = trailing_prefill(messages)
        if prefill is not None:
            raise NotImplementedError("Prefill continuation unsupported over chat-completions.")
        resp = _with_retries(
            lambda: self._client.chat.completions.create(
                model=self.model_id,
                messages=base_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                stop=stop,
                extra_body={"reasoning": {"enabled": False}},
            )
        )
        return [c.message.content or "" for c in resp.choices]


def _with_retries(fn, retries: int = 5, base_delay: float = 2.0):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - surface only after retries
            last = exc
            time.sleep(base_delay * (2 ** attempt))
    raise last
