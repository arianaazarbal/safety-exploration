"""Provider construction from the model registry.

Resolves a logical model name (from ``config/models.yaml``, either the ``models``
or ``aux`` section) into a concrete provider instance. Local providers are cached
process-wide because loading Gemma weights is expensive; an experiment that
sweeps over conditions reuses one engine.
"""
from __future__ import annotations

from typing import Any

from ..config import Config, load_models
from ..logging_utils import get_logger
from ..usage import GLOBAL_USAGE
from .anthropic_provider import AnthropicProvider
from .base import ChatProvider
from .openai_compat import OPENROUTER_BASE_URL, OpenAICompatProvider

log = get_logger("providers.registry")

_LOCAL_CACHE: dict[tuple, ChatProvider] = {}


def _resolve_entry(name: str, models_cfg: Config) -> dict:
    models = models_cfg.to_dict().get("models", {})
    aux = models_cfg.to_dict().get("aux", {})
    if name in models:
        return {"name": name, **models[name]}
    if name in aux:
        entry = dict(aux[name])
        # aux entries may reference a target model (e.g. calm_data_generator).
        if "ref" in entry:
            ref = entry["ref"]
            return _resolve_entry(ref, models_cfg)
        return {"name": name, **entry}
    raise KeyError(f"Unknown model {name!r} (not in models.yaml models/ or aux/)")


def build_provider(
    name: str,
    models_cfg: Config | None = None,
    run_cfg: Config | None = None,
    *,
    usage=None,
    prefer_local_backend: str = "vllm",
    adapter: str | None = None,
    require_capability: str | None = None,
) -> ChatProvider:
    models_cfg = models_cfg or load_models()
    usage = usage or GLOBAL_USAGE
    retry_cfg = run_cfg.retry.to_dict() if run_cfg and "retry" in run_cfg else {}

    entry = _resolve_entry(name, models_cfg)
    provider = entry["provider"]

    # Probing/logits forces the transformers backend regardless of preference.
    if require_capability == "logits":
        prefer_local_backend = "transformers"

    if provider == "anthropic":
        return AnthropicProvider(name, entry["api_id"], retry_cfg=retry_cfg, usage=usage)

    if provider == "openai":
        return OpenAICompatProvider(
            name, entry["api_id"], retry_cfg=retry_cfg, usage=usage,
            api_key_env="OPENAI_API_KEY",
        )

    if provider == "openrouter":
        return OpenAICompatProvider(
            name, entry["api_id"], retry_cfg=retry_cfg, usage=usage,
            base_url=OPENROUTER_BASE_URL, api_key_env="OPENROUTER_API_KEY",
            disable_thinking=bool(entry.get("disable_thinking", False)),
        )

    if provider == "local_hf":
        adapter = adapter if adapter is not None else entry.get("adapter")
        cache_key = (name, adapter, prefer_local_backend)
        if cache_key in _LOCAL_CACHE:
            return _LOCAL_CACHE[cache_key]
        kwargs: dict[str, Any] = dict(
            model=name,
            model_id=entry["hf_id"],
            retry_cfg=retry_cfg,
            usage=usage,
            is_instruct=bool(entry.get("is_instruct", True)),
            adapter=adapter,
        )
        if prefer_local_backend == "vllm":
            from .local_hf import VLLMProvider

            prov: ChatProvider = VLLMProvider(**kwargs)
        else:
            from .local_hf import TransformersProvider

            prov = TransformersProvider(**kwargs)
        _LOCAL_CACHE[cache_key] = prov
        return prov

    raise ValueError(f"Unknown provider {provider!r} for model {name!r}")


# Backwards-compatible alias.
get_provider = build_provider
