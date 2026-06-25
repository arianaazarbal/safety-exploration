"""Model backends.

Two concrete backends implement the same :class:`ModelBackend` interface:

* :class:`~gnh.models.hf_backend.HFBackend` -- local Gemma inference + LoRA.
* :class:`~gnh.models.openrouter_backend.OpenRouterBackend` -- Gemini via the
  OpenAI-compatible OpenRouter API.

Claude instruments (judge, auditor, paraphraser) use the thin Anthropic client
in :mod:`gnh.models.anthropic_client`.
"""

from gnh.models.base import Message, ModelBackend, get_backend

__all__ = ["Message", "ModelBackend", "get_backend"]
