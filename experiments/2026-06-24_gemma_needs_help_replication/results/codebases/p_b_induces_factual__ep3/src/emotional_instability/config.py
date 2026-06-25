"""Configuration loading.

A single ``config.yaml`` at the repo root drives every experiment. We load it
into a lightweight nested object so call sites can write ``cfg.training.dpo.beta``
rather than threading dictionaries everywhere. Unknown keys are preserved, so the
YAML can grow without touching this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/emotional_instability/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class Config:
    """Dot-accessible, dict-backed config node.

    ``cfg.foo`` returns nested ``Config`` objects for mappings and raw values
    otherwise. ``cfg["foo"]`` and ``cfg.get("foo", default)`` also work.
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._wrap(self._data[name])
        except KeyError as exc:  # pragma: no cover - surfaced to caller
            raise AttributeError(
                f"config key {name!r} not found (available: {sorted(self._data)})"
            ) from exc

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._wrap(self._data[key]) if key in self._data else default

    def keys(self):
        return self._data.keys()

    def to_dict(self) -> dict[str, Any]:
        return self._data

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        return value

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self._data!r})"


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the YAML config, defaulting to ``config.yaml`` at the repo root.

    The ``EMO_CONFIG`` environment variable overrides the default path so the
    same scripts can target alternate configs (e.g. a reduced smoke-test run).
    """
    if path is None:
        path = os.environ.get("EMO_CONFIG", DEFAULT_CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)
