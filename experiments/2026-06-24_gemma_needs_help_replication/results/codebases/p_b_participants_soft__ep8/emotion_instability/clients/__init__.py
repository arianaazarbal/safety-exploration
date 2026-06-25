"""Model client abstraction over local HF, OpenRouter and Anthropic backends."""
from .base import ChatClient, GenConfig, Message
from .factory import get_client

__all__ = ["ChatClient", "GenConfig", "Message", "get_client"]
