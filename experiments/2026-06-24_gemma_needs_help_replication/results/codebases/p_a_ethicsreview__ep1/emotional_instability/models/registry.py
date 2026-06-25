"""Factories that turn a :class:`ModelSpec` into a live client.

Kept separate from the client modules so that constructing a spec (cheap, no
heavy imports) is decoupled from instantiating a client (loads a 27B model or
opens an API connection).
"""

from __future__ import annotations

from ..config import ModelSpec
from .base import ChatClient


def build_target_client(spec: ModelSpec, *, adapter_path: str | None = None,
                        **kwargs) -> ChatClient:
    """Instantiate a target model client (Gemma local / Gemini API)."""
    if spec.backend == "gemma":
        from .gemma import GemmaClient

        return GemmaClient(
            spec.identifier,
            role=spec.role or "instruct",
            adapter_path=adapter_path,
            **kwargs,
        )
    if spec.backend == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(spec.identifier, **kwargs)
    raise ValueError(f"Unknown target backend: {spec.backend!r}")


def build_infra_client(spec: ModelSpec):
    """Instantiate an infrastructure model client (Anthropic / OpenAI)."""
    if spec.backend == "anthropic":
        from .api_clients import AnthropicClient

        return AnthropicClient(spec.identifier)
    if spec.backend == "openai":
        from .api_clients import OpenAIClient

        return OpenAIClient(spec.identifier)
    raise ValueError(f"Unknown infra backend: {spec.backend!r}")
