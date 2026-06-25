"""Configuration loading.

Loads ``config/default.yaml`` into a lightweight dotted-access object. We keep
this deliberately simple (no pydantic) so the config stays readable and easy to
override from scripts or environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class Config(dict):
    """A dict with attribute access that recurses into nested dicts/lists."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return _wrap(value)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, key: str) -> Path:
        """Resolve a path entry from the ``paths`` section, relative to repo root."""
        rel = self["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def model_entry(cfg: Config, name: str) -> Config:
    """Return the registry entry for ``name`` (raises if unknown)."""
    models = cfg["models"]
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(models)}")
    entry = Config(dict(models[name]))
    entry["name"] = name
    return entry


def ensure_dirs(cfg: Config) -> None:
    """Create all configured output directories."""
    for key in cfg["paths"]:
        cfg.get_path(key).mkdir(parents=True, exist_ok=True)
