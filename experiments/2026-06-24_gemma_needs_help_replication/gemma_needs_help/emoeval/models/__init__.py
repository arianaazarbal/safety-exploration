"""Unified inference interface over local Gemma and API-based Gemini models."""
from .base import ChatModel, Message
from .loader import load_model

__all__ = ["ChatModel", "Message", "load_model"]
