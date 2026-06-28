"""Adapter for any OpenAI-compatible endpoint (local vLLM/Ollama, Together,
Groq, etc.). Thin wrapper over OpenAIAdapter with a required base_url."""

from __future__ import annotations

from typing import Any

from .base import AdapterError
from .openai import OpenAIAdapter


class OpenAICompatAdapter(OpenAIAdapter):
    def __init__(
        self,
        model: str,
        api_key: str | None,
        params: dict[str, Any] | None = None,
        id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not base_url:
            raise AdapterError("openai_compat adapter requires a base_url")
        super().__init__(
            model=model, api_key=api_key, params=params, id=id, base_url=base_url
        )
