"""OpenRouter backend (OpenAI-compatible) — used for Gemini targets and the
GPT-5-mini judge cross-check, as in the paper.

Auth: set OPENROUTER_API_KEY. Base URL defaults to OpenRouter; override with
OPENROUTER_BASE_URL (e.g. to point at the Google API directly).
"""
from __future__ import annotations

import os
from typing import Sequence

from ..config import ModelSpec
from .base import Message


class OpenRouterClient:
    def __init__(self, spec: ModelSpec):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai is required for the openrouter backend") from e
        self.spec = spec
        self.name = spec.name
        self.model_id = spec.openrouter_id or spec.model
        if not self.model_id:
            raise ValueError(f"Model '{spec.name}' has no openrouter_id")
        self._client = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )

    def _extra_body(self) -> dict:
        body: dict = {}
        if self.spec.disable_thinking:
            # Paper: "we set thinking to be false via the API." OpenRouter exposes
            # this via the `reasoning` field; Google models accept reasoning effort.
            body["reasoning"] = {"enabled": False}
        return body

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> str:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self._extra_body() or None,
        )
        return resp.choices[0].message.content or ""
