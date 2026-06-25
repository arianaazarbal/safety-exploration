"""Model factory: turn a registry name into a live client.

Clients are cached per (name, force_backend) so repeated lookups within a process
reuse loaded local weights. ``force_backend='openrouter'`` lets Section 2 eval run
Gemma over the API when local GPUs are unavailable (the paper ran Gemma locally,
but the eval itself only needs ``chat``).
"""
from __future__ import annotations

from functools import lru_cache

from ..config import Config, model_entry
from .base import ModelClient
from .hf_local import HFLocalClient
from .openrouter import OpenRouterClient

_CACHE: dict[tuple[str, str | None], ModelClient] = {}


def get_client(cfg: Config, name: str, force_backend: str | None = None) -> ModelClient:
    cache_key = (name, force_backend)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    entry = model_entry(cfg, name)
    backend = force_backend or entry["backend"]

    if backend == "openrouter":
        model_id = entry.get("openrouter_id") or entry["model_id"]
        client: ModelClient = OpenRouterClient(
            name=name,
            model_id=model_id,
            base_url=cfg.openrouter.base_url,
            api_key_env=cfg.openrouter.api_key_env,
            max_retries=cfg.openrouter.max_retries,
            timeout_s=cfg.openrouter.timeout_s,
            disable_thinking=cfg.disable_thinking,
        )
    elif backend == "hf":
        client = HFLocalClient(
            name=name,
            model_id=entry["model_id"],
            dtype=cfg.local.dtype,
            device=cfg.local.device,
            load_in_4bit=cfg.local.load_in_4bit,
            adapter_path=entry.get("adapter_path"),
            hf_token_env=cfg.local.hf_token_env,
            cache_dir=cfg.local.cache_dir,
            max_new_tokens=cfg.local.max_new_tokens,
        )
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"Unknown backend '{backend}' for model '{name}'")

    _CACHE[cache_key] = client
    return client


def is_base_model(cfg: Config, name: str) -> bool:
    return bool(model_entry(cfg, name).get("is_base", False))
