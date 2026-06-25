"""Gemini backend via OpenRouter (OpenAI-compatible API).

The paper accesses Gemini through OpenRouter and disables "thinking" via the API
(noting Pro may still emit hidden reasoning). We mirror that here. Sampling ``n``
completions is done with sequential requests for portability, since not all
OpenRouter-proxied providers honour the ``n`` parameter.
"""
from __future__ import annotations

import time
from typing import Sequence

import config
from .base import ChatModel, Message


class GeminiChatModel(ChatModel):
    def __init__(self, name: str, *, max_retries: int = 4):
        self.name = name
        self.model_id = config.API_MODELS[name]
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not config.API.openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set.")
            self._client = OpenAI(
                api_key=config.API.openrouter_api_key,
                base_url=config.API.openrouter_base_url,
            )
        return self._client

    def _extra_body(self) -> dict:
        """Provider-specific knobs: turn off Gemini's thinking budget."""
        if not config.DISABLE_THINKING:
            return {}
        # OpenRouter passes `reasoning` through to Google; a zero budget disables
        # extended thinking where the provider supports it.
        return {"reasoning": {"enabled": False}}

    def _one_completion(self, messages, temperature, max_new_tokens) -> str:
        payload = [m.as_dict() for m in messages]
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    extra_body=self._extra_body(),
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - broad retry on transient API errors
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini call failed after {self.max_retries} retries: {last_err}")

    def _generate(self, messages, temperature, max_new_tokens, n):
        return [
            self._one_completion(messages, temperature, max_new_tokens)
            for _ in range(n)
        ]
