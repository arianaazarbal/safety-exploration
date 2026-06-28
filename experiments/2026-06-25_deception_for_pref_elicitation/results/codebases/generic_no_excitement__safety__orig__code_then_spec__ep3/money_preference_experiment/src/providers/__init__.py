"""Provider registry.

Add a new backend by implementing LLMProvider (see base.py) and registering it
in `_REGISTRY` below.
"""

from __future__ import annotations

from .base import (
    Block,
    Completion,
    InferenceSettings,
    LLMProvider,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)


def _make_anthropic(model: str, settings: InferenceSettings) -> LLMProvider:
    from .anthropic_provider import AnthropicProvider

    return AnthropicProvider(model, settings)


def _make_unimplemented(name: str):
    def _factory(model: str, settings: InferenceSettings) -> LLMProvider:
        raise NotImplementedError(
            f"The '{name}' provider is not implemented. Implement LLMProvider in "
            f"src/providers/{name}_provider.py and register it in src/providers/__init__.py. "
            f"The harness keeps the conversation in a provider-neutral block format, so a "
            f"new provider only needs to translate to/from its own wire format."
        )

    return _factory


# provider name -> factory(model, settings) -> LLMProvider
_REGISTRY = {
    "anthropic": _make_anthropic,
    # Stubs — fill these in to test other vendors. Kept as explicit, named
    # entries so config validation fails loudly rather than silently.
    "openai": _make_unimplemented("openai"),
    "google": _make_unimplemented("google"),
}


def build_provider(provider: str, model: str, settings: InferenceSettings) -> LLMProvider:
    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown provider {provider!r}. Known providers: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[provider](model, settings)


__all__ = [
    "Block",
    "Completion",
    "InferenceSettings",
    "LLMProvider",
    "Message",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "Usage",
    "build_provider",
]
