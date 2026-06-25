"""Pluggable model backends.

A backend turns a chat-formatted conversation into a sampled assistant
completion. Two implementations are provided:

- :class:`~emotional_instability.models.hf_backend.HFBackend` for local Gemma
  inference via HuggingFace ``transformers`` (supports response prefilling and
  exposes activations/logits for Appendix I).
- :class:`~emotional_instability.models.openrouter_backend.OpenRouterBackend`
  for Gemini via the OpenRouter API (no prefill, no internals).

Use :func:`~emotional_instability.models.registry.load_backend` to construct the
right backend for a registered model name.
"""

from .base import ChatMessage, GenerationResult, ModelBackend

__all__ = ["ChatMessage", "GenerationResult", "ModelBackend"]
