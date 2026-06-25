"""Unified chat-model interface and backends."""

from .base import ChatModel, Message, GenerationResult
from .registry import build_model, build_judge, get_spec

__all__ = ["ChatModel", "Message", "GenerationResult", "build_model",
           "build_judge", "get_spec"]
