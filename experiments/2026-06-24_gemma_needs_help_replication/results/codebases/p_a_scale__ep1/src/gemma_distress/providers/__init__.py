"""Model provider abstraction.

A provider turns a list of chat messages into generated text. The interface is
deliberately narrow so that API and local-inference backends are interchangeable
for the experiments that only need generation (Section 2, Petri). Capabilities
beyond plain chat (assistant prefill, logits) are optional and declared per
provider; experiments that need them assert the capability up front.
"""
from .base import (
    ChatMessage,
    ChatProvider,
    GenerationResult,
    PrefillCapable,
    LogitsCapable,
)
from .registry import build_provider, get_provider

__all__ = [
    "ChatMessage",
    "ChatProvider",
    "GenerationResult",
    "PrefillCapable",
    "LogitsCapable",
    "build_provider",
    "get_provider",
]
