"""Model-access layer.

Three backends, one interface (:class:`base.ModelClient`):
  * ``hf``         — local HuggingFace inference for the open-weight Gemma
                     models. Supports chat, raw prefilled continuation, and
                     hidden-state capture (needed for Section 3 prefill and the
                     Appendix I probing).
  * ``openrouter`` — OpenAI-compatible API for Gemini participants and the
                     GPT-5-mini cross-judge.
  * ``anthropic``  — Claude judges/auditors (Sonnet 4, Opus 4).

Closed models (Gemini) only support :meth:`chat`; prefill/hidden-state methods
raise ``NotImplementedError`` there, matching the paper's noted limitation that
its prefill and probing experiments cannot be run on Gemini.
"""

from .base import ChatMessage, GenerationResult, ModelClient
from .factory import get_client

__all__ = ["ChatMessage", "GenerationResult", "ModelClient", "get_client"]
