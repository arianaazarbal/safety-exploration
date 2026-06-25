"""Model client abstractions and registry."""
from .base import ChatMessage, GenerationResult, ModelClient
from .registry import get_client, list_targets

__all__ = [
    "ChatMessage",
    "GenerationResult",
    "ModelClient",
    "get_client",
    "list_targets",
]
