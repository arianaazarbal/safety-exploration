"""Model client abstractions.

A `ChatClient` exposes a uniform `generate` interface over a list of
chat-format messages, regardless of whether the model runs locally
(HuggingFace / vLLM) or behind an API (OpenRouter for Gemini).

Base (pretrained) models additionally support `continue_text`, which appends
to a raw string prefill rather than completing a chat turn -- this is what the
Section 3 prefill experiments need.
"""
from .base import ChatClient, Message, GenResult
from .factory import build_client, build_finetuned_client

__all__ = ["ChatClient", "Message", "GenResult", "build_client",
           "build_finetuned_client"]
