"""Model adapter registry."""

from __future__ import annotations

from typing import Any

from .base import (
    ModelClient,
    ModelSession,
    ModelTurn,
    ToolCall,
    ToolResult,
    ToolSpec,
)

_PROVIDERS = {
    "anthropic": ("grantbench.models.anthropic_client", "AnthropicClient"),
    "openai": ("grantbench.models.openai_client", "OpenAIClient"),
}


def build_client(provider: str, model: str, params: dict[str, Any] | None = None) -> ModelClient:
    """Instantiate the adapter for a provider."""
    import importlib

    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Known: {sorted(_PROVIDERS)}")
    module_path, cls_name = _PROVIDERS[provider]
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(model=model, params=params or {})


__all__ = [
    "ModelClient",
    "ModelSession",
    "ModelTurn",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "build_client",
]
