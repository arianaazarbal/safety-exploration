"""Map a ModelConfig to a concrete adapter instance."""

from __future__ import annotations

from ..config import ModelConfig
from .base import ModelAdapter


def build_adapter(mc: ModelConfig, *, max_output_tokens: int) -> ModelAdapter:
    provider = mc.provider.lower()

    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(
            label=mc.label,
            model_id=mc.model,
            max_output_tokens=max_output_tokens,
            effort=mc.effort or "high",
        )

    if provider == "openai":
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(
            label=mc.label,
            model_id=mc.model,
            max_output_tokens=max_output_tokens,
        )

    if provider == "local":
        # OpenAI-compatible local endpoint (vLLM/Ollama/etc.).
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(
            label=mc.label,
            model_id=mc.model,
            max_output_tokens=max_output_tokens,
            base_url=mc.base_url,
            api_key=mc.extra.get("api_key", "not-needed"),
        )

    if provider == "google":
        from .google_adapter import GoogleAdapter

        return GoogleAdapter(
            label=mc.label,
            model_id=mc.model,
            max_output_tokens=max_output_tokens,
        )

    raise ValueError(f"unknown provider: {mc.provider!r}")
