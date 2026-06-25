"""Config loading and the in-scope model registry."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"


@dataclass
class ModelSpec:
    name: str
    kind: str          # gemma_vllm | gemini
    chat: bool
    role: str          # instruct | base
    family: str        # gemma | gemini
    hf_id: str | None = None
    api_id: str | None = None
    # For finetuned variants: an adapter path layered on top of `base_model`.
    adapter_path: str | None = None
    base_model: str | None = None


class Config:
    """Thin wrapper over config.yaml with a resolved model registry."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.models: dict[str, ModelSpec] = {}
        for name, m in raw.get("models", {}).items():
            self.models[name] = ModelSpec(
                name=name,
                kind=m["kind"],
                chat=m.get("chat", True),
                role=m.get("role", "instruct"),
                family=m.get("family", "unknown"),
                hf_id=m.get("hf_id"),
                api_id=m.get("api_id"),
            )

    # ---- convenience accessors -------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("output_dir", "./runs"))

    def section(self, key: str) -> dict[str, Any]:
        return self.raw.get(key, {})

    def model(self, name: str) -> ModelSpec:
        if name in self.models:
            return self.models[name]
        # Finetuned variant syntax: "<base>+<tag>" -> adapter under output_dir.
        if "+" in name:
            base, tag = name.split("+", 1)
            base_spec = self.models[base]
            adapter = self.output_dir / "training" / f"{base}.{tag}"
            return ModelSpec(
                name=name, kind=base_spec.kind, chat=base_spec.chat,
                role=base_spec.role, family=base_spec.family, hf_id=base_spec.hf_id,
                base_model=base, adapter_path=str(adapter),
            )
        raise KeyError(f"Unknown model: {name}")


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else _DEFAULT_CONFIG
    with open(path) as f:
        return Config(yaml.safe_load(f))
