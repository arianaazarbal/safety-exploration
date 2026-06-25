"""Typed access to ``config.yaml``.

We keep the on-disk config as plain YAML (easy to diff / override) and expose it
through a thin dataclass wrapper so the rest of the codebase gets attribute
access and a single place to resolve model specs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass(frozen=True)
class ModelSpec:
    """How to instantiate a single model client."""

    name: str
    kind: str  # "local_hf" | "openrouter"
    family: str
    is_instruct: bool = True
    hf_id: str | None = None
    api_id: str | None = None
    adapter_path: str | None = None  # LoRA adapter to load on top of hf_id

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ModelSpec":
        return cls(
            name=name,
            kind=d["kind"],
            family=d.get("family", "unknown"),
            is_instruct=d.get("is_instruct", True),
            hf_id=d.get("hf_id"),
            api_id=d.get("api_id"),
            adapter_path=d.get("adapter_path"),
        )


@dataclass
class Config:
    """Whole-config wrapper. ``raw`` holds the parsed YAML dict; helper methods
    surface the parts used most often with light validation."""

    raw: dict[str, Any]
    path: Path

    # --- model resolution -------------------------------------------------
    def model_spec(self, name: str) -> ModelSpec:
        models = self.raw["models"]
        if name not in models:
            raise KeyError(
                f"Unknown model '{name}'. Known models: {sorted(models)}"
            )
        return ModelSpec.from_dict(name, models[name])

    @property
    def eval_models(self) -> list[str]:
        return list(self.raw["eval_models"])

    # --- convenience getters ---------------------------------------------
    def section(self, *keys: str) -> Any:
        node: Any = self.raw
        for k in keys:
            node = node[k]
        return node

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["runtime"]["output_dir"])

    @property
    def seed(self) -> int:
        return int(self.raw["runtime"]["seed"])

    def scaled_response_counts(self) -> dict[str, int]:
        """Per-category response budgets after applying ``sampling.scale``.

        Used by smoke-test runs to shrink a 4000-response sweep to ~tens of
        responses without changing the protocol.
        """
        scale = float(self.raw["sampling"]["scale"])
        out: dict[str, int] = {}
        for cat, n in self.raw["sampling"]["responses_per_model"].items():
            out[cat] = max(1, round(n * scale))
        return out


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=cfg_path)
