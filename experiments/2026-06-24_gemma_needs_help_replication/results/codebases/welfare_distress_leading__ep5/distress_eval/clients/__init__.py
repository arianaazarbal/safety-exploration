"""Client registry: build a ChatClient from a (backend, model_id) pair.

Backends are imported lazily inside `build_client` so that, e.g., running with
only OpenRouter configured does not require the anthropic / google / torch
packages to be installed.
"""

from __future__ import annotations

from .base import ChatClient, Message  # re-export

_OPENAI_COMPAT = {"openrouter", "openai", "openai_compat"}


def build_client(backend: str, model_id: str) -> ChatClient:
    backend = backend.lower()
    if backend in _OPENAI_COMPAT:
        from .openai_compat import OpenAICompatClient

        return OpenAICompatClient(model_id, backend=backend)
    if backend in {"google", "google_genai", "gemini"}:
        from .google_genai import GoogleGenAIClient

        return GoogleGenAIClient(model_id)
    if backend == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model_id)
    if backend in {"hf", "hf_local", "transformers"}:
        from .hf_local import HFLocalClient

        return HFLocalClient(model_id)
    raise ValueError(f"Unknown backend: {backend!r}")


__all__ = ["ChatClient", "Message", "build_client"]
