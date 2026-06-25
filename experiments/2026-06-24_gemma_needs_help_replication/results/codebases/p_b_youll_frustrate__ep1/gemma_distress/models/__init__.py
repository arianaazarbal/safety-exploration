"""Model client abstraction and backends.

``load_model(spec)`` returns a :class:`ChatModel` for either the Gemini API or
a local HuggingFace checkpoint. All target-model generation in the harness goes
through this interface so the rollout / prefill code is backend-agnostic.
"""
from __future__ import annotations

from ..config import ModelSpec
from .base import ChatModel, Message


def load_model(spec: ModelSpec) -> ChatModel:
    if spec.backend == "gemini":
        from .gemini import GeminiModel

        return GeminiModel(spec)
    if spec.backend == "hf":
        from .hf_local import HFLocalModel

        return HFLocalModel(spec)
    raise ValueError(f"unknown backend: {spec.backend!r}")


__all__ = ["ChatModel", "Message", "load_model"]
