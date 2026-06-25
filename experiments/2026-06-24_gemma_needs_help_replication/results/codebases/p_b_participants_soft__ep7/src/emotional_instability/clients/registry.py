"""Factory that maps a model name -> a concrete ModelClient, honouring backend
routing rules (cloud vs local) and optional overrides.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import ModelSpec, load_config
from .base import ModelClient
from .local_hf import LocalHFClient
from .openrouter import OpenRouterClient


def _build(spec: ModelSpec, prefer_local: bool, thinking: bool) -> ModelClient:
    backend = spec.backend

    if backend == "local_adapter":
        # Finetuned participant: load base weights + LoRA adapter locally.
        cfg = load_config()
        base = cfg.model(spec.base_model)
        return LocalHFClient(
            model_name=spec.name,
            hf_id=base.hf_id,
            is_instruct=True,
            adapter_path=spec.adapter_path,
        )

    if backend == "local" or (prefer_local and spec.hf_id):
        return LocalHFClient(
            model_name=spec.name,
            hf_id=spec.hf_id,
            is_instruct=spec.is_instruct,
        )

    if backend == "openrouter":
        return OpenRouterClient(
            model_name=spec.name, openrouter_id=spec.openrouter_id, thinking=thinking
        )

    raise ValueError(f"No client route for {spec.name!r} (backend={backend})")


@lru_cache(maxsize=None)
def get_client(name: str, prefer_local: bool = False, thinking: bool = False) -> ModelClient:
    """Return a (cached) client for the named model.

    prefer_local: force HF local inference even when an OpenRouter id exists
      (used for Section 3 where prefill continuation must be exact, and for any
      offline run). Has no effect on Gemini (no local weights).
    """
    cfg = load_config()
    spec = cfg.model(name)
    return _build(spec, prefer_local=prefer_local, thinking=thinking)
