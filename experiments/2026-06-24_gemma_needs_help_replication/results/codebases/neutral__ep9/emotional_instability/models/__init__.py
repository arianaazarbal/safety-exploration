"""Model backends: a thin uniform interface over local HF and hosted models."""
from .base import ChatMessage, GenerationConfig, ModelBackend
from .registry import get_backend, clear_backend_cache

__all__ = [
    "ChatMessage",
    "GenerationConfig",
    "ModelBackend",
    "get_backend",
    "clear_backend_cache",
]
