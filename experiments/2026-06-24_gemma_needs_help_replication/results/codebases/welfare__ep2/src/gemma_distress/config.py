"""Configuration loading and shared path helpers.

All experiment scripts read the YAML files under ``config/`` through these
helpers so that paths and model definitions live in exactly one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/gemma_distress/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
OUTPUT_DIR = Path(os.environ.get("GEMMA_DISTRESS_OUTPUT", REPO_ROOT / "outputs"))


def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_models() -> dict[str, Any]:
    return _load_yaml("models.yaml")


def load_eval() -> dict[str, Any]:
    return _load_yaml("eval.yaml")


def load_training() -> dict[str, Any]:
    return _load_yaml("training.yaml")


@dataclass
class ModelSpec:
    """Resolved entry from ``models.yaml`` targets."""

    name: str
    family: str
    backend: str          # hf_local | openrouter
    kind: str             # instruct | base
    hf_id: str | None = None
    api_id: str | None = None
    adapter_path: str | None = None
    chat_template: bool = True
    disable_thinking: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_registry(cls, name: str, entry: dict[str, Any]) -> "ModelSpec":
        known = {
            "family", "backend", "kind", "hf_id", "api_id",
            "adapter_path", "chat_template", "disable_thinking",
        }
        return cls(
            name=name,
            family=entry["family"],
            backend=entry["backend"],
            kind=entry.get("kind", "instruct"),
            hf_id=entry.get("hf_id"),
            api_id=entry.get("api_id"),
            adapter_path=entry.get("adapter_path"),
            chat_template=entry.get("chat_template", True),
            disable_thinking=entry.get("disable_thinking", False),
            extra={k: v for k, v in entry.items() if k not in known},
        )


def get_model_spec(name: str) -> ModelSpec:
    reg = load_models()["targets"]
    if name not in reg:
        raise KeyError(
            f"Unknown model '{name}'. Known targets: {sorted(reg)}"
        )
    return ModelSpec.from_registry(name, reg[name])


def output_path(*parts: str) -> Path:
    """Build (and create) a path under the output directory."""
    p = OUTPUT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
