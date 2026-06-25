"""Configuration loading and access.

Configs are plain YAML. We support layering a base config (``default.yaml``)
with an optional user override file and ad-hoc ``key.subkey=value`` overrides
from the CLI, so that scaled runs can be parameterised without editing files.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Repo layout: <repo>/config/*.yaml and <repo>/src/gemma_distress/config.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"
DEFAULT_MODELS_PATH = _REPO_ROOT / "config" / "models.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce_scalar(value: str) -> Any:
    """Best-effort parse of a CLI string override into a Python scalar."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _apply_dot_override(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override into non-mapping at {p!r} in {dotted_key!r}")
    node[parts[-1]] = value


class Config:
    """Attribute/dict hybrid view over a nested config mapping.

    ``cfg.sampling.temperature`` and ``cfg["sampling"]["temperature"]`` both work.
    Unknown keys raise ``KeyError`` rather than returning ``None`` so that typos
    in long-running jobs fail fast instead of silently mis-configuring a sweep.
    """

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        try:
            val = self._data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(val) if isinstance(val, dict) else val

    def __getitem__(self, item: str) -> Any:
        val = self._data[item]
        return Config(val) if isinstance(val, dict) else val

    def get(self, item: str, default: Any = None) -> Any:
        val = self._data.get(item, default)
        return Config(val) if isinstance(val, dict) else val

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self._data!r})"


@dataclass
class LoadedConfig:
    run: Config
    models: Config
    raw: dict


def load_config(
    config_path: str | os.PathLike | None = None,
    models_path: str | os.PathLike | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load run config, applying overrides of the form ``a.b.c=value``."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override {ov!r} must be of the form key.subkey=value")
        key, _, raw_value = ov.partition("=")
        _apply_dot_override(data, key.strip(), _coerce_scalar(raw_value.strip()))

    return Config(data)


def load_models(models_path: str | os.PathLike | None = None) -> Config:
    path = Path(models_path) if models_path else DEFAULT_MODELS_PATH
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Config(data)
