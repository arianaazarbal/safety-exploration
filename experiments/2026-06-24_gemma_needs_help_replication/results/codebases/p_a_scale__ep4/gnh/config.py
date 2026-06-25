"""Typed configuration loaded from YAML.

We keep the schema permissive (extra keys allowed) so the YAML can carry
documentation and experiment-specific knobs without breaking older code, but
the fields the code actually depends on are validated.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


class ProviderConfig(_Base):
    kind: str
    base_url: str | None = None
    api_key_env: str | None = None
    max_concurrency: int = 8
    requests_per_minute: int = 0  # 0 == unlimited
    timeout_s: float = 180.0
    max_retries: int = 8

    def api_key(self) -> str:
        """Resolve the API key from the environment at call time (never stored)."""
        if not self.api_key_env:
            return ""
        return os.environ.get(self.api_key_env, "")


class ModelConfig(_Base):
    provider: str
    api_model: str
    hf_id: str | None = None
    adapter_path: str | None = None
    family: str = "unknown"
    kind: str = "chat"  # "chat" | "base"
    role: str = "target"  # "target" | "tool"
    supports_prefill: bool = False
    disable_thinking: bool = False
    chat_template_source: str = "hf"


class RunConfig(_Base):
    output_dir: str = "runs"
    seed: int = 0
    max_concurrency: int = 32
    log_level: str = "INFO"


class Config(_Base):
    run: RunConfig = Field(default_factory=RunConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    target_models: list[str] = Field(default_factory=list)
    finetune_models: list[str] = Field(default_factory=list)
    eval: dict[str, Any] = Field(default_factory=dict)
    prefill: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    petri: dict[str, Any] = Field(default_factory=dict)
    benchmarks: dict[str, Any] = Field(default_factory=dict)
    probing: dict[str, Any] = Field(default_factory=dict)

    # ---- convenience accessors -------------------------------------------------
    def model(self, name: str) -> ModelConfig:
        if name not in self.models:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self.models)}")
        return self.models[name]

    def provider_for(self, model_name: str) -> ProviderConfig:
        prov = self.model(model_name).provider
        if prov not in self.providers:
            raise KeyError(f"Model '{model_name}' references unknown provider '{prov}'")
        return self.providers[prov]

    @property
    def output_path(self) -> Path:
        p = Path(self.run.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
