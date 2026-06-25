"""Inference backends.

The eval/training code depends only on the abstract ``ModelBackend`` interface
in ``base.py``. Concrete backends:

    hf_backend.HFBackend       - local HuggingFace Gemma (chat + prefill + logits)
    api_backend.OpenRouterBackend - Gemini targets via OpenRouter
    api_backend.AnthropicBackend  - Claude judges / Petri auditor & judge
    api_backend.OpenAIBackend     - GPT-5-mini judge cross-check

Use ``build_backend(model_id)`` to get the right backend for a model id.
"""

from .base import ModelBackend, GenerationResult, build_backend

__all__ = ["ModelBackend", "GenerationResult", "build_backend"]
