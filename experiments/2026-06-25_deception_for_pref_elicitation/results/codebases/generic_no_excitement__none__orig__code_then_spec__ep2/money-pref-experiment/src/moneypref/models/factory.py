"""Build a ModelClient from a subject/auditor spec."""

from __future__ import annotations

from typing import Any

from .base import ModelClient


def build_client(spec: dict[str, Any]) -> ModelClient:
    """`spec` is a dict like {id, provider, effort?}. Returns a fresh, unstarted client."""
    provider = spec.get("provider", "anthropic")
    model_id = spec["id"]

    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(
            model_id,
            effort=spec.get("effort", "high"),
            max_tokens=spec.get("max_tokens", 16000),
            thinking=spec.get("thinking", True),
        )
    if provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(model_id)

    raise ValueError(f"unknown provider: {provider!r}")
