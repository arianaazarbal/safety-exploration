"""Typed configuration loaded from the YAML files in ``config/``.

Pydantic models give us validation-at-load (fail fast at the start of a multi-week run
rather than 6 hours in) and a single source of truth for defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


class BackendConfig(BaseModel):
    kind: Literal["openai_compat"] = "openai_compat"
    base_url: str
    api_key_env: str
    timeout_s: float = 180.0
    max_concurrency: int = 8
    default_extra_headers: dict[str, str] = Field(default_factory=dict)

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            # vLLM accepts any non-empty token; emit a harmless placeholder so local
            # runs do not require the user to export a dummy var.
            return "EMPTY"
        return key


class ModelConfig(BaseModel):
    backend: str
    model_id: str
    family: str
    role: Literal["instruct", "base"] = "instruct"
    chat: bool = True
    supports_prefill: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class JudgeConfig(BaseModel):
    backend: str
    model_id: str
    pinned_paper_id: str | None = None
    temperature: float = 0.0


class ModelsConfig(BaseModel):
    backends: dict[str, BackendConfig]
    models: dict[str, ModelConfig]
    judges: dict[str, JudgeConfig]

    def model(self, name: str) -> ModelConfig:
        if name not in self.models:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self.models)}")
        return self.models[name]

    def backend_for(self, model_or_judge: ModelConfig | JudgeConfig) -> BackendConfig:
        return self.backends[model_or_judge.backend]


def load_models_config(path: Path | str | None = None) -> ModelsConfig:
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "models.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ModelsConfig.model_validate(raw)


def load_experiments_config(path: Path | str | None = None) -> dict[str, Any]:
    """Experiments config stays a plain dict: it is read by many small scripts that each
    want different slices, and over-typing it adds friction without much safety."""
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "experiments.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
