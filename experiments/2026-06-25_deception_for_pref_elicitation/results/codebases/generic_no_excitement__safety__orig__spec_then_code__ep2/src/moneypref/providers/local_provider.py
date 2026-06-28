"""Local / self-hosted backend.

Any OpenAI-compatible server (vLLM, Ollama's OpenAI shim, text-generation-inference,
LM Studio, etc.) works by pointing ``base_url`` at it. This is a thin alias over the
OpenAI provider with a forgiving default key, since local servers usually ignore it.
"""

from __future__ import annotations

from typing import Any

from .openai_provider import OpenAIProvider


class LocalProvider(OpenAIProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", **kw: Any) -> None:
        super().__init__(model=model, base_url=base_url, **kw)
        self.label = f"local:{model}"
