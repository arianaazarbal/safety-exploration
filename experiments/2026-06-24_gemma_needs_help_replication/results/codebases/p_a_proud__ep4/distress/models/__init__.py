"""Model client implementations and a registry-backed factory.

All target models, judges, and auditors are accessed through the ``ChatModel``
interface (see ``base.py``). Backends:

    HFBackend         local Gemma via HuggingFace Transformers (chat + prefill +
                      hidden-state extraction)
    OpenRouterBackend OpenAI-compatible HTTP API (Gemini targets)
    OpenAIBackend     OpenAI-compatible HTTP API (GPT-5-mini judge validation)
    AnthropicBackend  Anthropic SDK (Claude judge / Petri auditor & judge)
"""

from .base import ChatModel, GenerationError
from .factory import build_model

__all__ = ["ChatModel", "GenerationError", "build_model"]
