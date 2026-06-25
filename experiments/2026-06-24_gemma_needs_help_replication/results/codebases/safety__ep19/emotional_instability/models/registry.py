"""Build :class:`ModelClient` instances from the YAML model registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import ModelClient

_DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def load_model_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    path = Path(path) if path else _DEFAULT_REGISTRY
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["models"]


def build_model(key: str, registry: dict[str, dict[str, Any]] | None = None) -> ModelClient:
    """Instantiate the model identified by ``key`` in the registry.

    Backends are imported lazily so that, e.g., running an API-only eval does
    not require torch/transformers to be installed.
    """
    registry = registry or load_model_registry()
    if key not in registry:
        raise KeyError(f"Unknown model '{key}'. Known: {sorted(registry)}")
    spec = dict(registry[key])
    backend = spec.pop("backend")
    spec.setdefault("name", key)

    if backend == "hf":
        from .hf_model import HFModel

        return HFModel(**spec)
    if backend == "openrouter":
        from .api_model import OpenRouterModel

        return OpenRouterModel(**spec)
    if backend == "anthropic":
        from .api_model import AnthropicModel

        return AnthropicModel(**spec)
    raise ValueError(f"Unknown backend '{backend}' for model '{key}'.")
