"""Resolve a model key to a concrete backend.

Keys in scope:
  * gemma-3-{27b,12b}-{it,pt}  -> local HF backend (GemmaLocalModel)
  * gemini-2.5-{flash,pro}     -> OpenRouter backend (GeminiOpenRouterModel)

An optional ``adapter_path`` layers a LoRA finetune (Section 4) on a Gemma key.
"""

from __future__ import annotations

from emotional_stability.config import (
    GEMINI_API_MODELS,
    GEMMA_LOCAL_MODELS,
    Settings,
)
from emotional_stability.models.base import ChatModel
from emotional_stability.records import Message


def _fold_system(messages: list[Message]) -> list[Message]:
    """Fold a leading system message into the first user turn (Gemma has no
    system role). No-op if there is no system message."""
    if not messages or messages[0].role != "system":
        return messages
    system = messages[0]
    rest = messages[1:]
    for i, m in enumerate(rest):
        if m.role == "user":
            merged = Message(role="user", content=f"{system.content}\n\n{m.content}")
            return rest[:i] + [merged] + rest[i + 1 :]
    # No user turn: demote system to a user turn.
    return [Message(role="user", content=system.content)] + rest


def get_chat_model(
    key: str,
    *,
    adapter_path: str | None = None,
    settings: Settings | None = None,
    **kwargs,
) -> ChatModel:
    if key in GEMMA_LOCAL_MODELS or adapter_path is not None:
        from emotional_stability.models.gemma import GemmaLocalModel

        return GemmaLocalModel(
            key, adapter_path=adapter_path, settings=settings, **kwargs
        )
    if key in GEMINI_API_MODELS:
        from emotional_stability.models.gemini import GeminiOpenRouterModel

        return GeminiOpenRouterModel(key, settings=settings)
    raise ValueError(
        f"model key {key!r} is out of scope for this replication "
        f"(in scope: {sorted(GEMMA_LOCAL_MODELS) + sorted(GEMINI_API_MODELS)})"
    )
