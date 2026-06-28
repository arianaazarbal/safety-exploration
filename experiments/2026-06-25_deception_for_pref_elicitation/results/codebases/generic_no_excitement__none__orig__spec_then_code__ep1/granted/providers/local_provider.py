"""Local / open-weights adapter via an OpenAI-compatible endpoint (vLLM, Ollama,
LM Studio, etc.).

Reuses the OpenAI wire format but points at ``GRANTED_LOCAL_BASE_URL``.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_provider import OpenAIProvider


class LocalProvider(OpenAIProvider):
    def __init__(self, model: str, **kwargs: Any) -> None:
        # Skip OpenAIProvider.__init__ (which builds a default client); build a
        # client pointed at the local endpoint instead.
        from openai import OpenAI

        # Provider.__init__
        super(OpenAIProvider, self).__init__(model, **kwargs)
        base_url = kwargs.get("base_url") or os.environ.get(
            "GRANTED_LOCAL_BASE_URL", "http://localhost:8000/v1"
        )
        api_key = kwargs.get("api_key") or os.environ.get(
            "GRANTED_LOCAL_API_KEY", "not-needed"
        )
        self._client = OpenAI(base_url=base_url, api_key=api_key)
