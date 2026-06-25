"""Build a runnable model object from its registry name."""
from __future__ import annotations

from ..config import ModelRegistry, ModelSpec
from .base import GenerationConfig


def gen_config_for(spec: ModelSpec, **overrides) -> GenerationConfig:
    base = dict(
        max_new_tokens=spec.max_new_tokens,
        temperature=spec.temperature,
        top_p=spec.top_p,
    )
    base.update({k: v for k, v in overrides.items() if v is not None})
    return GenerationConfig(**base)


def build_model(name: str, registry: ModelRegistry | None = None):
    """Instantiate the backend object for `name`. Heavy deps imported lazily."""
    registry = registry or ModelRegistry.load()
    spec = registry.get(name)
    if spec.backend == "hf_local":
        from .hf_local import load_hf_model

        return load_hf_model(spec)
    if spec.backend == "openrouter":
        from .api import OpenRouterModel

        return OpenRouterModel(spec)
    if spec.backend == "anthropic":
        from .api import AnthropicModel

        return AnthropicModel(spec)
    raise ValueError(f"Unknown backend '{spec.backend}' for model '{name}'.")
