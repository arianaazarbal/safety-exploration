"""OpenAI client used only for the GPT-5-mini secondary judge in the judge-agreement
check (Section 2.1). Reads ``OPENAI_API_KEY``.

Uses the Responses API (``client.responses.create``) which is the current surface for
GPT-5-family models; falls back to chat.completions if the SDK predates it.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import Message, ChatModel
from config import MAX_API_RETRIES, API_BACKOFF_BASE


class OpenAIClient:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY for the GPT-5-mini secondary judge")
        self._client = OpenAI()

    def complete(
        self,
        messages: list[Message],
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self._ensure_client()
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages += [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("system", "user", "assistant")
        ]

        def _call():
            if hasattr(self._client, "responses"):
                resp = self._client.responses.create(
                    model=self.model_id,
                    input=api_messages,
                    max_output_tokens=max_tokens,
                )
                return (getattr(resp, "output_text", None) or "").strip()
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        return ChatModel._retry(_call, retries=MAX_API_RETRIES, base=API_BACKOFF_BASE)
