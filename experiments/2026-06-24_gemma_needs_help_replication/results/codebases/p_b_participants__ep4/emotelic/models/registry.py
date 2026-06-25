"""Build an LLMClient from a ModelSpec, merging in config defaults.

Local HF models are cached process-wide so that, e.g., the DPO-eval and the
vanilla-eval can share a base model load, and so the judge isn't re-instantiated
per call.
"""
from __future__ import annotations

from functools import lru_cache

from emotelic.config import ModelSpec, ModelsConfig, load_models
from emotelic.models.base import LLMClient

_CACHE: dict[str, LLMClient] = {}


def _instantiate(spec: ModelSpec, defaults: dict) -> LLMClient:
    p = {**defaults, **spec.params}
    if spec.backend == "openrouter":
        from emotelic.models.openrouter import OpenRouterClient

        return OpenRouterClient(spec.name, or_id=p["or_id"], thinking=p.get("thinking", False))
    if spec.backend == "anthropic":
        from emotelic.models.anthropic_client import AnthropicClient

        return AnthropicClient(spec.name, model=p["model"])
    if spec.backend == "hf_local":
        from emotelic.models.hf_local import HFLocalClient

        return HFLocalClient(
            spec.name,
            hf_id=p["hf_id"],
            is_instruct=p.get("is_instruct", True),
            adapter_path=p.get("adapter_path"),
            load_in_4bit=p.get("load_in_4bit", False),
        )
    raise ValueError(f"Unknown backend '{spec.backend}' for model '{spec.name}'")


def build_client(name: str, models_cfg: ModelsConfig | None = None) -> LLMClient:
    if name in _CACHE:
        return _CACHE[name]
    cfg = models_cfg or load_models()
    spec = cfg.get(name)
    client = _instantiate(spec, cfg.defaults)
    _CACHE[name] = client
    return client


@lru_cache(maxsize=1)
def _default_models() -> ModelsConfig:
    return load_models()
