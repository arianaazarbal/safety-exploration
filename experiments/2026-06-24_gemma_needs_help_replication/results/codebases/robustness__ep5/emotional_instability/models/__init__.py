"""Model-client implementations and a factory keyed by `ModelSpec.kind`."""
from __future__ import annotations

from .base import ChatMessage, ModelClient


def make_client(spec, **kwargs) -> "ModelClient":
    """Instantiate the right client for a `config.ModelSpec`."""
    if spec.kind == "hf":
        from .hf_model import HFModelClient
        return HFModelClient(spec, **kwargs)
    if spec.kind == "openrouter":
        from .gemini_model import OpenRouterClient
        return OpenRouterClient(spec, **kwargs)
    raise ValueError(f"Unknown model kind: {spec.kind!r}")


__all__ = ["ChatMessage", "ModelClient", "make_client"]
