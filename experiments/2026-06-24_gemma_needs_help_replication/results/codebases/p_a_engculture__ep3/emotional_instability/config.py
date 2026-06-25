"""Configuration loading.

A single YAML file (``config/default.yaml``) drives the whole pipeline. We keep
the loaded config as nested dataclasses for the parts that the orchestration code
touches frequently (models, sampling, judge), and expose the raw dict for the
long tail of experiment-specific knobs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    """A single addressable model (target, base, judge, or finetune)."""

    name: str
    backend: str                      # "vllm" | "hf" | "openrouter" | "anthropic"
    family: str = "unknown"           # "gemma" | "gemini" | "claude" | ...
    kind: str = "instruct"            # "instruct" | "base"
    hf_id: str | None = None          # HuggingFace id for local backends
    api_id: str | None = None         # provider id for API backends
    adapter_path: str | None = None   # LoRA adapter dir for finetuned variants

    @property
    def is_base(self) -> bool:
        return self.kind == "base"


@dataclass
class Config:
    raw: dict[str, Any]
    seed: int
    paths: dict[str, str]
    target_models: list[ModelSpec]
    base_models: list[ModelSpec]
    finetune_base: str
    judge: dict[str, Any]
    sampling: dict[str, Any]
    eval_conditions: dict[str, dict[str, Any]]

    # ---- convenience accessors for the less-touched sections -------------
    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    def output_path(self, *parts: str) -> Path:
        p = Path(self.paths.get("output_dir", "outputs")).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def data_path(self, *parts: str) -> Path:
        p = Path(self.paths.get("data_dir", "data_cache")).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def model_by_name(self, name: str) -> ModelSpec:
        for m in (*self.target_models, *self.base_models):
            if m.name == name:
                return m
        raise KeyError(f"Unknown model '{name}'. Known: "
                       f"{[m.name for m in (*self.target_models, *self.base_models)]}")


def _to_specs(items: list[dict[str, Any]]) -> list[ModelSpec]:
    return [ModelSpec(**it) for it in items]


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    path = Path(path or os.environ.get("EI_CONFIG", "config/default.yaml"))
    raw = yaml.safe_load(path.read_text())
    return Config(
        raw=raw,
        seed=raw.get("seed", 0),
        paths=raw.get("paths", {}),
        target_models=_to_specs(raw.get("target_models", [])),
        base_models=_to_specs(raw.get("base_models", [])),
        finetune_base=raw.get("finetune_base", "gemma-3-27b-it"),
        judge=raw.get("judge", {}),
        sampling=raw.get("sampling", {}),
        eval_conditions=raw.get("eval_conditions", {}),
    )
