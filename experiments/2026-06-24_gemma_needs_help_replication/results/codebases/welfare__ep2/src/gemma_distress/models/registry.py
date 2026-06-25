"""Factory that turns a registry name into a live ``ChatModel``."""
from __future__ import annotations

from ..config import ModelSpec, get_model_spec
from .base import ChatModel


def load_model(name: str, **backend_kwargs) -> ChatModel:
    """Instantiate the backend for a registry target.

    ``backend_kwargs`` are forwarded to the backend constructor (e.g.
    ``tensor_parallel_size`` for vLLM, ``max_workers`` for OpenRouter).
    """
    spec: ModelSpec = get_model_spec(name)

    if spec.backend == "hf_local":
        from .hf_local import HFLocalModel

        return HFLocalModel(
            name=spec.name,
            hf_id=spec.hf_id,
            is_base=(spec.kind == "base"),
            adapter_path=spec.adapter_path,
            **backend_kwargs,
        )

    if spec.backend == "openrouter":
        from .openrouter import OpenRouterModel

        return OpenRouterModel(
            name=spec.name,
            api_id=spec.api_id,
            disable_thinking=spec.disable_thinking,
            **backend_kwargs,
        )

    raise ValueError(f"Unknown backend '{spec.backend}' for model '{name}'")
