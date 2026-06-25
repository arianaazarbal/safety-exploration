"""Target-model clients (Gemma local, Gemini API) behind one interface."""

from .base import ChatMessage, ModelClient
from .factory import load_model

__all__ = ["ChatMessage", "ModelClient", "load_model"]
