"""Local / open-weight provider.

Most local serving stacks (vLLM, Ollama, LM Studio, TGI) expose an OpenAI-compatible
endpoint, so this is a thin subclass of OpenAIProvider pointed at a local base_url.
Defaults to http://localhost:8000/v1 if none is given.
"""
from __future__ import annotations

from typing import Any

from .openai_provider import OpenAIProvider


class LocalProvider(OpenAIProvider):
    name = "local"

    def __init__(self, model: str, *, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(model, base_url=base_url or "http://localhost:8000/v1", **kwargs)
