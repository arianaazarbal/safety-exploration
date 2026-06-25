"""OpenRouter backend for the Gemini models (closed weights), via the
OpenAI-compatible Chat Completions endpoint.

Appendix B.1: thinking is disabled via the API where the provider allows it.
OpenRouter exposes a `reasoning` field; we set `{"enabled": false}` so Gemini
runs without visible reasoning. The paper notes Gemini 2.5 Pro may still emit
hidden reasoning that this flag does not suppress.
"""
from __future__ import annotations

import os
import time

from .. import config
from .base import ChatModel, GenerationConfig, Message


class OpenRouterChatModel(ChatModel):
    is_base = False

    def __init__(self, key: str, model_id: str):
        from openai import OpenAI

        api_key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {config.OPENROUTER_API_KEY_ENV} to use OpenRouter model {model_id}."
            )
        self.key = key
        self.model_id = model_id
        self.client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)

    def generate(self, messages, *, prefill=None, gen=None):
        if prefill is not None:
            # Section 3 prefilling is Gemma-only (base models); closed Gemini
            # cannot be prefilled or studied as a base model (paper limitation).
            raise NotImplementedError(
                "Prefilled continuation is not supported for closed Gemini models."
            )
        gen = gen or GenerationConfig()
        extra_body = {}
        if config.DISABLE_THINKING:
            extra_body["reasoning"] = {"enabled": False}

        last_err = None
        for attempt in range(config.API_MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    max_tokens=gen.max_new_tokens,
                    extra_body=extra_body or None,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # rate limits / transient 5xx
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")
