"""Provider registry.

Only the Anthropic provider is implemented. Other providers are explicit
NotImplementedError stubs so the runner cannot silently mis-handle a model whose
adapter doesn't exist yet — each needs its own faithful translation of messages,
tool calls, and (ideally) thinking-state preservation.
"""

from __future__ import annotations

from ..config import MODEL_REGISTRY
from .anthropic_provider import AnthropicModel
from .base import (
    LanguageModel,
    Message,
    ModelTurn,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__all__ = [
    "LanguageModel",
    "Message",
    "ModelTurn",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "build_model",
]


def build_model(model_key: str, *, max_tokens: int) -> LanguageModel:
    """Instantiate the LanguageModel for a registry key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model key {model_key!r}")
    provider, model_id = MODEL_REGISTRY[model_key]
    if provider == "anthropic":
        return AnthropicModel(model_id, max_tokens=max_tokens)
    raise NotImplementedError(
        f"provider {provider!r} has no adapter yet. Implement a LanguageModel for it "
        f"(faithful message/tool translation) before testing model {model_key!r}."
    )
