"""Build a `ModelClient` from a `ModelSpec` + config."""
from __future__ import annotations

from ..config import Config, ModelSpec
from .api_model import APIModelClient
from .base import ModelClient
from .hf_model import HFModelClient


def build_client(spec: ModelSpec, cfg: Config, lora_path: str | None = None) -> ModelClient:
    gen_cfg = cfg.get("generation", {})
    if spec.backend == "hf":
        if not spec.hf_id:
            raise ValueError(f"Model {spec.name} has backend 'hf' but no hf_id")
        return HFModelClient(
            name=spec.name, hf_id=spec.hf_id, hf_cfg=cfg.get("hf", {}),
            generation_cfg=gen_cfg, lora_path=lora_path,
        )
    if spec.backend == "api":
        if not spec.api_id:
            raise ValueError(f"Model {spec.name} has backend 'api' but no api_id")
        if lora_path:
            raise ValueError("LoRA adapters are not applicable to API models")
        return APIModelClient(
            name=spec.name, api_id=spec.api_id, api_cfg=cfg.get("api", {}),
            generation_cfg=gen_cfg, disable_thinking=spec.disable_thinking,
        )
    raise ValueError(f"Unknown backend '{spec.backend}' for model {spec.name}")
