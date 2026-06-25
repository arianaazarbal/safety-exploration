"""Unified inference clients.

The rest of the codebase talks to models exclusively through
:class:`~emotional_instability.clients.base.ModelClient`. Use
:func:`build_client` to construct the right implementation from a
:class:`~emotional_instability.config.ModelSpec`.
"""

from __future__ import annotations

from ..config import ModelSpec
from .base import ChatMessage, GenerationConfig, ModelClient


def build_client(spec: ModelSpec, **kwargs) -> ModelClient:
    """Instantiate the client implementation named by ``spec.backend``.

    Imports are deferred so that, e.g., running a pure-API Gemini eval does not
    require torch/transformers to be installed.
    """
    backend = spec.backend
    if backend == "openrouter":
        from .openrouter import OpenRouterClient

        return OpenRouterClient(spec, **kwargs)
    if backend == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(spec, **kwargs)
    if backend == "hf":
        from .hf_local import HuggingFaceClient

        return HuggingFaceClient(spec, **kwargs)
    if backend == "vllm":
        try:
            from .vllm_local import VLLMClient

            return VLLMClient(spec, **kwargs)
        except Exception as exc:  # pragma: no cover - optional dependency
            # vLLM is an optional accelerator; the HF client produces identical
            # samples (same weights), just slower. Fall back transparently.
            import logging

            logging.getLogger(__name__).warning(
                "vLLM backend unavailable (%s); falling back to HuggingFace.", exc
            )
            from .hf_local import HuggingFaceClient

            return HuggingFaceClient(spec, **kwargs)
    raise ValueError(f"Unknown backend '{backend}' for model '{spec.name}'.")


__all__ = [
    "ChatMessage",
    "GenerationConfig",
    "ModelClient",
    "build_client",
]
