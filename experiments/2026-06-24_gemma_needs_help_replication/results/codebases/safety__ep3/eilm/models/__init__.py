"""Model backends and a registry to construct them from a ModelSpec."""

from .base import ChatModel, Message
from .registry import get_model

__all__ = ["ChatModel", "Message", "get_model"]
