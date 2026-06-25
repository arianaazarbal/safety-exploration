"""Configuration loading.

Loads config/config.yaml into a dict-like object with attribute access, resolves
paths relative to the configured root, and loads a .env for API keys. Kept
deliberately simple (no schema framework) so the YAML stays the single source of
truth that DESIGN.md documents.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional
    load_dotenv = None

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


class Config:
    """Thin attribute/dict wrapper around the parsed YAML."""

    def __init__(self, data: Dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    # --- path resolution -----------------------------------------------------
    def path(self, name: str) -> Path:
        """Resolve a named path from the `paths` block, relative to root."""
        raw = self._data["paths"][name]
        p = Path(raw)
        if not p.is_absolute():
            p = self.root / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def raw(self) -> Dict[str, Any]:
        return self._data


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    repo_root = cfg_path.resolve().parent.parent
    declared_root = Path(data.get("paths", {}).get("root", "."))
    root = declared_root if declared_root.is_absolute() else (repo_root / declared_root).resolve()

    # Load .env from repo root if present, so API keys are available.
    if load_dotenv is not None:
        env_file = repo_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    return Config(data, root)


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is required but not set. "
            f"Add it to the environment or a .env file at the repo root."
        )
    return val
