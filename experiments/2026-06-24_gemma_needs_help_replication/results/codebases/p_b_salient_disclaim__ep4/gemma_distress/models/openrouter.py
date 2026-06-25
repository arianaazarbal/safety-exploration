"""OpenRouter backend for API-hosted targets (the Gemini family).

The paper accesses Gemini-2.5-Flash / Gemini-2.5-Pro through OpenRouter
(Appendix B.1) and sets thinking to false via the API. OpenRouter is
OpenAI-compatible, so we use the ``openai`` SDK pointed at the OpenRouter base
URL. Thinking is disabled with the OpenRouter ``reasoning`` extra-body field;
note Appendix B.1's caveat that "Gemini-2.5 Pro ... may produce hidden reasoning
that is not prevented by this setting".

This same client also serves the GPT-5-mini judge-reliability cross-check
(Section 2.1), which the paper likewise reaches as an API model.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

from ..config import (DISABLE_THINKING, OPENROUTER_API_KEY_ENV,
                      OPENROUTER_BASE_URL)
from .base import Message, ModelClient


class OpenRouterClient(ModelClient):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        disable_thinking: bool = DISABLE_THINKING,
        max_retries: int = 5,
        base_url: str = OPENROUTER_BASE_URL,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries
        self.base_url = base_url
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
            if not api_key:
                raise RuntimeError(f"{OPENROUTER_API_KEY_ENV} is not set")
            self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        return self._client

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        if prefill is not None:
            # Gemini is a black-box target only; the paper never prefills it.
            raise NotImplementedError("OpenRouter targets do not support prefill")
        client = self._ensure_client()
        extra_body = {}
        if self.disable_thinking:
            # OpenRouter normalises this across providers; for Gemini it maps to
            # disabling the thinking budget where the provider honours it.
            extra_body["reasoning"] = {"enabled": False}

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    extra_body=extra_body or None,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # transient API errors -> exponential backoff
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")
