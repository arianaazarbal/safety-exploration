"""Config loading + lightweight typed accessors.

The whole pipeline is driven by ``config.yaml`` at the repo root. We keep the
loader intentionally dumb (a nested dict wrapped in ``Config``) so that adding a
knob to the YAML never requires touching a dataclass.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class Config:
    """Dot/standard-access wrapper around the parsed YAML config."""

    def __init__(self, data: dict[str, Any], root: Path = REPO_ROOT):
        self._data = data
        self.root = root

    # --- generic access ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    # --- path helpers (resolved relative to repo root, created on demand) --
    def path(self, key: str) -> Path:
        p = self.root / self._data["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def results_dir(self) -> Path:
        return self.path("results_dir")

    @property
    def cache_dir(self) -> Path:
        return self.path("cache_dir")

    @property
    def adapters_dir(self) -> Path:
        return self.path("adapters_dir")


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data, root=path.resolve().parent)
