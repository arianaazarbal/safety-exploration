"""Model client factory."""
from __future__ import annotations

from typing import Any

from ..config import load_models_config
from .base import Conversation, GenParams, Message, ModelClient, ModelSpec


def build_client(spec: ModelSpec, adapter_path: str | None = None, **kwargs: Any) -> ModelClient:
    backend = spec.backend
    if backend == "openai":
        from .api_model import OpenAICompatClient

        return OpenAICompatClient(spec)
    if backend == "anthropic":
        from .api_model import AnthropicClient

        return AnthropicClient(spec)
    if backend == "hf":
        from .hf_model import HFModelClient

        return HFModelClient(spec, adapter_path=adapter_path, **kwargs)
    if backend == "vllm":
        from .vllm_model import VLLMModelClient

        return VLLMModelClient(spec, adapter_path=adapter_path, **kwargs)
    raise ValueError(f"Unknown backend: {backend!r}")


def get_target_spec(name: str, models_cfg: dict[str, Any] | None = None) -> ModelSpec:
    models_cfg = models_cfg or load_models_config()
    if name not in models_cfg.get("targets", {}):
        raise KeyError(f"Target {name!r} not in models.yaml targets.")
    return ModelSpec.from_dict(name, models_cfg["targets"][name])


def get_role_spec(role_path: str, models_cfg: dict[str, Any] | None = None) -> ModelSpec:
    """Fetch a non-target model spec by dotted path, e.g. 'judges.primary'."""
    models_cfg = models_cfg or load_models_config()
    node: Any = models_cfg
    for part in role_path.split("."):
        node = node[part]
    return ModelSpec.from_dict(role_path, node)


def build_target(name: str, adapter_path: str | None = None, **kwargs: Any) -> ModelClient:
    return build_client(get_target_spec(name), adapter_path=adapter_path, **kwargs)


def build_role(role_path: str, **kwargs: Any) -> ModelClient:
    return build_client(get_role_spec(role_path), **kwargs)


__all__ = [
    "Conversation",
    "GenParams",
    "Message",
    "ModelClient",
    "ModelSpec",
    "build_client",
    "build_target",
    "build_role",
    "get_target_spec",
    "get_role_spec",
]
