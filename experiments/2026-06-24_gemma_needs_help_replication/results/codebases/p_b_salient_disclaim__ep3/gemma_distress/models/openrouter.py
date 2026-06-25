"""OpenRouter backend (OpenAI-compatible) for Gemini targets.

Reads OPENROUTER_API_KEY. Thinking is requested off via the `reasoning`
extra-body field (Appendix B.1: "we set thinking to be false via the API.
However, Gemini-2.5 Pro ... may produce hidden reasoning that is not prevented
by this setting.").

Gemini is closed-source: no prefill, no hidden states. Those experiments
(Sections 3 / Appendix I) are therefore Gemma-only, as in the paper.
"""

from __future__ import annotations

import os
import time

from config import ModelSpec
from .base import ChatModel, Message


class OpenRouterChatModel(ChatModel):
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, spec: ModelSpec, *, disable_thinking: bool = True,
                 max_retries: int = 5):
        from openai import OpenAI

        self.spec = spec
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries
        self.client = OpenAI(
            base_url=self.BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def generate(self, messages, *, max_new_tokens=None, temperature=None,
                 n=1, prefill=None):
        from config import MAX_NEW_TOKENS, TEMPERATURE
        if prefill is not None:
            raise NotImplementedError(
                "Gemini (OpenRouter) does not support assistant prefilling; "
                "prefill experiments are Gemma-only."
            )
        max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        temperature = TEMPERATURE if temperature is None else temperature

        extra_body = {}
        if self.disable_thinking:
            # OpenRouter normalises reasoning controls across providers.
            extra_body["reasoning"] = {"enabled": False}

        completions: list[str] = []
        for _ in range(n):
            completions.append(
                self._one(messages, max_new_tokens, temperature, extra_body)
            )
        return completions

    def _one(self, messages, max_tokens, temperature, extra_body) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.openrouter_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_exc}")
