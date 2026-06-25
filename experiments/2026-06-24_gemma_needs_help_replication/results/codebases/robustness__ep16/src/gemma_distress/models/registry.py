"""Build a :class:`ChatModel` from a config :class:`ModelSpec`.

Local models are cached per-process (loading a 27B checkpoint is expensive), so
repeated ``build_model`` calls for the same name reuse the loaded weights.
"""

from __future__ import annotations

from ..config import Config, ModelSpec
from .base import ChatModel

_LOCAL_CACHE: dict[str, ChatModel] = {}


def build_model(spec: ModelSpec, cfg: Config, reuse_local: bool = True) -> ChatModel:
    if spec.kind == "openrouter":
        from .openrouter import OpenRouterModel

        return OpenRouterModel(
            name=spec.name,
            api_id=spec.api_id,
            family=spec.family,
            is_instruct=spec.is_instruct,
            max_retries=cfg.get("runtime", "api_max_retries", default=6),
        )

    if spec.kind == "local_hf":
        if reuse_local and spec.name in _LOCAL_CACHE:
            return _LOCAL_CACHE[spec.name]
        from .local_hf import LocalHFModel

        local_cfg = cfg.get("runtime", "local", default={}) or {}
        model = LocalHFModel(
            name=spec.name,
            hf_id=spec.hf_id,
            family=spec.family,
            is_instruct=spec.is_instruct,
            dtype=local_cfg.get("dtype", "bfloat16"),
            device_map=local_cfg.get("device_map", "auto"),
            load_in_4bit=local_cfg.get("load_in_4bit", False),
            adapter_path=spec.adapter_path,
        )
        if reuse_local:
            _LOCAL_CACHE[spec.name] = model
        return model

    raise ValueError(f"Unknown model kind: {spec.kind!r}")


def build_by_name(name: str, cfg: Config, **kwargs) -> ChatModel:
    return build_model(cfg.model_spec(name), cfg, **kwargs)
