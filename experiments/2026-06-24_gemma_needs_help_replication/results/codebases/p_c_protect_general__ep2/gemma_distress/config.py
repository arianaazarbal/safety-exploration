"""Configuration loading.

A thin wrapper over the YAML config that gives attribute-style access and merges
CLI / programmatic overrides. We deliberately keep this dependency-light (a dotted
dict) rather than a rigid schema, because the experiments are research code that we
expect to poke at from scripts.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class Config(dict):
    """Dict with attribute access and recursive wrapping of nested dicts."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    """Load the YAML config, applying an optional dict of deep overrides."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if overrides:
        data = _deep_update(copy.deepcopy(data), overrides)
    return Config(data)


def model_spec(cfg: Config, name: str) -> Config:
    """Return the spec block for a configured model, erroring clearly if missing."""
    models = cfg.get("models", {})
    if name not in models:
        raise KeyError(
            f"Model '{name}' is not configured. Known models: {sorted(models)}"
        )
    spec = dict(models[name])
    spec.setdefault("name", name)
    return Config(spec)
