"""OpenAI client.

Used only for the cross-judge reliability validation (GPT-5-mini re-scoring of
260 sampled responses, Section 2.1). Kept minimal.
"""
from __future__ import annotations

import os
from typing import Sequence

from .base import ChatMessage, GenerationResult, ModelClient


class OpenAIClient(ModelClient):
    def __init__(self, spec: dict):
        self.key = spec.get("key", spec["model_id"])
        self.model_id = spec["model_id"]
        self.supports_prefill = False
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict] | None = None,
    ) -> GenerationResult:
        self._ensure_client()
        api_messages = [m.as_dict() for m in messages]
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=list(stop) if stop else None,
        )
        text = resp.choices[0].message.content or ""
        return GenerationResult(
            text=text, stop_reason=resp.choices[0].finish_reason, raw=resp)
