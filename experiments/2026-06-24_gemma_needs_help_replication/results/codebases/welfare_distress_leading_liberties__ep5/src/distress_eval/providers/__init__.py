"""Provider factory: build a ChatModel from a config spec."""

from __future__ import annotations

from .base import ChatModel


def build_model(spec: dict) -> ChatModel:
    """Construct a ChatModel from a `{id, provider, ...}` spec dict.

    Providers:
      - "google"    : Gemma-3-*-it and Gemini-2.5-* via google-genai
      - "anthropic" : Claude judge
      - "local"     : Gemma via local vLLM/transformers (extra kwargs passed through)
    """
    provider = spec.get("provider", "google")
    model_id = spec["id"]

    if provider == "google":
        from .google import GoogleChatModel

        return GoogleChatModel(model_id, api_key=spec.get("api_key"))
    if provider == "anthropic":
        from .anthropic import AnthropicChatModel

        return AnthropicChatModel(model_id, api_key=spec.get("api_key"))
    if provider == "local":
        from .local_hf import LocalHFChatModel

        return LocalHFChatModel(
            model_id,
            engine=spec.get("engine", "vllm"),
            dtype=spec.get("dtype", "bfloat16"),
            tensor_parallel_size=spec.get("tensor_parallel_size", 1),
            max_model_len=spec.get("max_model_len"),
        )
    raise ValueError(f"Unknown provider {provider!r} for model {model_id!r}")


__all__ = ["ChatModel", "build_model"]
