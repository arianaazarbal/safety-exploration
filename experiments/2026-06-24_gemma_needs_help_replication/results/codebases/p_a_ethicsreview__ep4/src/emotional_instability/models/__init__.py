"""Model backends and registry.

Two inference paths are supported (see ``config/models.yaml``):

* :class:`~emotional_instability.models.hf_backend.HFChatModel` -- local
  HuggingFace weights. Supports assistant *prefilling* (needed for base models
  and the Section 3 continuation experiment) and exposes the tokenizer /
  hidden-state access used by the Appendix I probing code.
* :class:`~emotional_instability.models.openrouter_backend.OpenRouterChatModel`
  -- Gemini via the OpenAI-compatible OpenRouter API. No prefill / no activations.

The Anthropic client (:mod:`~emotional_instability.models.anthropic_backend`)
powers the judge, paraphraser, onset labeller and Petri auditor/judge.
"""

from .base import ChatModel, Message
from .registry import load_model, model_info

__all__ = ["ChatModel", "Message", "load_model", "model_info"]
