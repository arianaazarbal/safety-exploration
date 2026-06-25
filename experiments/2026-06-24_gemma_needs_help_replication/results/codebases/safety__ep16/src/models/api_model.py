"""OpenRouter-backed chat client for the Gemini targets.

The paper accesses Gemini through OpenRouter (App. B.1) with the slugs
``google/gemini-2.5-flash`` and ``google/gemini-2.5-pro``, and sets thinking to
false. OpenRouter exposes an OpenAI-compatible API, so we reuse the ``openai``
SDK pointed at the OpenRouter base URL.

Reasoning/thinking is disabled via OpenRouter's ``reasoning={"enabled": False}``
extra-body field; note the paper's caveat that Gemini-2.5-Pro may still produce
hidden reasoning the flag does not suppress.
"""

from __future__ import annotations

import time

from config import API, DISABLE_THINKING
from src.models.base import ChatModel, Message


class OpenRouterChatModel(ChatModel):
    def __init__(self, name: str, model_id: str, *, max_retries: int = 5):
        self.name = name
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key = API.openrouter_api_key
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set; required for Gemini targets.")
            self._client = OpenAI(api_key=key, base_url=API.openrouter_base_url)
        return self._client

    def generate(self, messages: list[Message], *, temperature=1.0, top_p=1.0, max_new_tokens=2048, seed=None) -> str:
        extra_body: dict = {}
        if DISABLE_THINKING:
            # OpenRouter unifies provider reasoning controls under `reasoning`.
            extra_body["reasoning"] = {"enabled": False}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    seed=seed,
                    extra_body=extra_body or None,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # network / rate-limit / provider error
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} retries: {last_err}")
