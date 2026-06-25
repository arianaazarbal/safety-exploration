"""Model clients used across the replication.

A single small interface (:class:`~gemma_distress.models.base.ChatModel`) covers
both locally-hosted Gemma checkpoints and the API-served Gemini / Claude / GPT
models, so the eval harness, prefilling study, and training-data generators can
all treat models uniformly.
"""

from .base import ChatModel, Message, build_model

__all__ = ["ChatModel", "Message", "build_model"]
