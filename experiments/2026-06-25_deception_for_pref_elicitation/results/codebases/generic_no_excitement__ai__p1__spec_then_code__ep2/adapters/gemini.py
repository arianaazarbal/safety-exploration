"""Stub adapter for Google Gemini subject models.

Intentionally unimplemented in this build. To add support, translate the
neutral message/tool format (see adapters/base.py) to the google-genai API and
map function calls back into ToolCall.
"""

from __future__ import annotations

from .base import ModelAdapter, AdapterResponse


class GeminiAdapter(ModelAdapter):
    provider = "gemini"

    def complete(self, system: str, messages: list, tools: list) -> AdapterResponse:
        raise NotImplementedError(
            "GeminiAdapter is a stub. Implement neutral<->Gemini translation "
            "following the pattern in adapters/claude.py."
        )
