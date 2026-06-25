"""Construct the right :class:`ModelClient` for a :class:`ModelSpec`."""

from __future__ import annotations

from .base import ModelClient
from ..config import Config, ModelSpec


def build_client(
    spec: ModelSpec,
    cfg: Config | None = None,
    *,
    adapter_path: str | None = None,
    backend: str | None = None,
    **local_kwargs,
) -> ModelClient:
    """Build a client for ``spec``.

    ``adapter_path`` attaches a LoRA adapter (finetuned Gemma). ``backend``
    overrides the spec's default backend (e.g. force ``"hf"`` for prefill).
    """
    cfg = cfg or Config()
    backend = backend or spec.backend

    if backend == "openrouter":
        from .openrouter_client import OpenRouterChat

        return OpenRouterChat(
            spec,
            base_url=cfg.openrouter_base_url,
            api_key=cfg.openrouter_key(),
            max_retries=cfg.judge.max_retries,
        )

    if backend in ("vllm", "hf"):
        from .local import LocalChat

        return LocalChat(spec, backend=backend, adapter_path=adapter_path, **local_kwargs)

    raise ValueError(f"unknown backend {backend!r}")
