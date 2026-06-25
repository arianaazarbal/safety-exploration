"""Typed configuration loaded from config.yaml.

We keep this deliberately thin: dataclasses mirror the YAML structure so that
experiment code reads `cfg.training.dpo.beta` rather than dict-indexing. Unknown
keys are tolerated (forward-compatible) and surfaced via `.raw`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@dataclass
class ModelSpec:
    name: str
    kind: str                       # "hf" | "gemini"
    role: str = "instruct"          # "instruct" | "base"
    family: str = ""
    hf_id: str | None = None
    api_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Config:
    """Attribute-style view over the parsed YAML.

    Nested mappings become `Config` objects; lists/scalars pass through. This
    lets experiment modules use dotted access while keeping a single source of
    truth in config.yaml.
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Config({list(self._data)})"

    # -- model helpers -------------------------------------------------------
    def model_spec(self, name: str) -> ModelSpec:
        raw = self._data["models"][name]
        return ModelSpec(
            name=name,
            kind=raw["kind"],
            role=raw.get("role", "instruct"),
            family=raw.get("family", ""),
            hf_id=raw.get("hf_id"),
            api_id=raw.get("api_id"),
            raw=raw,
        )

    def output_root(self) -> Path:
        root = Path(self._data.get("output_dir", "runs"))
        if not root.is_absolute():
            root = REPO_ROOT / root
        return root


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return Config(data)
