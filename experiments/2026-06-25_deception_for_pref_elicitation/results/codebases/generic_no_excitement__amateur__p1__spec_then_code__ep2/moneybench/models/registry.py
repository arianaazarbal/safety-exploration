"""Adapter factory: turn a ModelRef into a concrete ModelAdapter."""

from __future__ import annotations

from ..config import ModelRef
from .base import ModelAdapter


def build_adapter(ref: ModelRef, max_output_tokens: int = 16000) -> ModelAdapter:
    """Instantiate the adapter for `ref`.

    The 'openai' provider also serves arbitrary OpenAI-compatible endpoints; to
    target a self-hosted open-weights server, set OPENAI_BASE_URL in the
    environment (read by the adapter via the openai SDK) or extend this factory.
    """

    provider = ref.provider
    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(model=ref.model, max_output_tokens=max_output_tokens)

    if provider in ("openai", "openai-compatible", "openweights"):
        from .openai_compatible_adapter import OpenAICompatibleAdapter

        import os

        return OpenAICompatibleAdapter(
            model=ref.model,
            max_output_tokens=max_output_tokens,
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )

    if provider == "google":
        from .google_adapter import GoogleAdapter

        return GoogleAdapter(model=ref.model, max_output_tokens=max_output_tokens)

    raise ValueError(
        f"unknown provider {provider!r}; expected one of: "
        f"anthropic, openai, openai-compatible, openweights, google"
    )
