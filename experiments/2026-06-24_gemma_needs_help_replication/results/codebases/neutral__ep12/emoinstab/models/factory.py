"""Factory that resolves a model name (from config/models.yaml) to a client."""
from __future__ import annotations

from typing import Optional

from ..config import Settings
from .anthropic_client import AnthropicModel
from .base import ChatModel
from .hf_local import HFLocalModel
from .openrouter import OpenRouterModel


def build_client(name: str, settings: Settings, *,
                 adapter_path: Optional[str] = None) -> ChatModel:
    """Build a ChatModel for `name`, which must be a key under `local:` or
    `api:` in models.yaml. `adapter_path` loads a LoRA adapter (local only)."""
    models = settings.models
    if name in models.get("local", {}):
        spec = models["local"][name]
        return HFLocalModel(
            name=name,
            hf_id=spec["hf_id"],
            family=spec.get("family", "gemma"),
            role=spec.get("role", "instruct"),
            adapter_path=adapter_path,
        )
    if name in models.get("api", {}):
        spec = models["api"][name]
        return OpenRouterModel(
            name=name,
            slug=spec["slug"],
            family=spec.get("family", "gemini"),
            reasoning=spec.get("reasoning", False),
        )
    raise KeyError(f"Unknown model '{name}' (not in models.yaml local/api).")


def build_judge(role: str, settings: Settings):
    """Build a judge/auditor handle. `role` is a key under `judges:`."""
    spec = settings.models["judges"][role]
    provider = spec["provider"]
    if provider == "anthropic":
        return AnthropicModel(name=role, model=spec["model"])
    if provider == "openrouter":
        return OpenRouterModel(name=role, slug=spec["slug"], family="gpt",
                               reasoning=spec.get("reasoning", False),
                               cache_namespace=f"judge/{role}")
    raise ValueError(f"Unknown judge provider '{provider}'")
