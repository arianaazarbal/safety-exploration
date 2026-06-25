"""Build concrete backends from a :class:`~emotional_eval.config.ModelSpec`."""

from __future__ import annotations

from ..config import ModelSpec, Registry
from .base import GenerationConfig
from .hf_backend import HFBackend
from .openrouter_backend import OpenRouterBackend


def build_backend(
    spec: ModelSpec,
    registry: Registry,
    *,
    adapter_path: str | None = None,
    max_new_tokens: int | None = None,
):
    """Instantiate the backend for ``spec`` using registry-wide defaults."""
    defaults = registry.defaults
    config = GenerationConfig(
        temperature=float(defaults.get("temperature", 1.0)),
        max_new_tokens=int(max_new_tokens or defaults.get("max_new_tokens", 1024)),
        thinking=bool(defaults.get("thinking", False)),
    )

    if spec.backend == "hf":
        return HFBackend(
            name=spec.name,
            hf_id=spec.hf_id,
            is_base=spec.is_base,
            config=config,
            backend_cfg=registry.backends.get("hf", {}),
            adapter_path=adapter_path,
        )

    if spec.backend == "openrouter":
        be = registry.backends.get("openrouter", {})
        return OpenRouterBackend(
            name=spec.name,
            api_id=spec.api_id,
            config=config,
            api_key=registry.api_key("openrouter"),
            base_url=be.get("base_url", "https://openrouter.ai/api/v1"),
            extra_body=spec.extra_body,
            max_retries=int(be.get("max_retries", 5)),
        )

    raise ValueError(f"unsupported backend {spec.backend!r} for {spec.name}")
