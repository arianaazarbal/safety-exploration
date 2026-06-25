"""Map a :class:`config.ModelSpec` (or a Claude model id) to a live client."""

from __future__ import annotations

from typing import Optional

from .base import ChatModel


def load_model(spec, *, adapter_path: Optional[str] = None, **kwargs) -> ChatModel:
    """Instantiate the client for ``spec``.

    ``spec`` may be a :class:`config.ModelSpec` (target models) or a plain
    Claude model-id string (judge/auditor). Extra kwargs are forwarded to the
    underlying client (e.g. ``load_in_4bit=True`` for Gemma).
    """
    # Plain string => Claude infra model.
    if isinstance(spec, str):
        from .claude_client import ClaudeClient

        return ClaudeClient(spec)

    if spec.provider == "gemma_hf":
        from .gemma_client import GemmaClient

        return GemmaClient(
            spec.model_id,
            name=spec.name,
            is_base=(spec.role == "base"),
            adapter_path=adapter_path,
            **kwargs,
        )
    if spec.provider == "gemini_api":
        from .gemini_client import GeminiClient

        return GeminiClient(spec.model_id, name=spec.name)

    raise ValueError(f"Unknown provider: {spec.provider!r}")
