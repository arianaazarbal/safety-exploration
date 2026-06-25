"""Model client backends.

A single `ChatClient` interface abstracts over the three provider backends in
scope:
  * hf_local   -- local Gemma weights (transformers / vLLM). Supports prefilled
                  continuations, which are required for Sections 3, 4-recovery
                  and I.
  * openrouter -- closed Gemini models via the OpenAI-compatible API.
  * anthropic  -- Claude judge / Petri auditor & judge.

`build_client(model_entry)` constructs the right backend from a config entry
(see config/models.yaml).
"""

from .base import ChatClient, ChatMessage, GenerationResult, build_client

__all__ = ["ChatClient", "ChatMessage", "GenerationResult", "build_client"]
