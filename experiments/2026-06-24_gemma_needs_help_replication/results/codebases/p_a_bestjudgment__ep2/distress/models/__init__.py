"""Model client layer.

Three backends behind one :class:`~distress.models.base.ModelClient` interface:

* :class:`~distress.models.local.LocalChat` — Gemma weights via vLLM (fast,
  batched) or transformers (fallback, and required for prefill continuations
  and finetuned adapters).
* :class:`~distress.models.openrouter_client.OpenRouterChat` — Gemini
  generation and the GPT-5-mini cross-judge via OpenRouter (OpenAI-compatible).
* :class:`~distress.models.anthropic_client.AnthropicChat` — Claude judge,
  Petri auditor/judge, onset labelling, paraphrasing.

Use :func:`build_client` to construct the right client for a ``ModelSpec``.
"""

from .base import ModelClient, Message  # noqa: F401
from .registry import build_client  # noqa: F401
