"""Configuration loading and lightweight dotted-access helpers.

The config is a plain nested dict loaded from YAML (``configs/default.yaml`` by
default). We deliberately avoid heavy schema frameworks; ``Config`` just wraps a
dict with attribute and dotted-key access so call sites read cleanly
(``cfg.sampling.temperature``, ``cfg.get("eval.history_format")``).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Config:
    """Attribute/dotted access over a nested dict."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # -- access -----------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def keys(self):
        return self._data.keys()

    def items(self):
        for k, v in self._data.items():
            yield k, (Config(v) if isinstance(v, dict) else v)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self._data!r})"


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load a YAML config, falling back to the bundled default."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)
