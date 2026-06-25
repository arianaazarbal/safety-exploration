"""Configuration loading.

Configs live as YAML under ``configs/`` and are loaded into plain dataclasses /
dicts. We keep this deliberately thin: the YAML is the source of truth, and these
helpers just locate, parse, and lightly validate it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels up from this file (distress/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class ModelSpec:
    """A single entry from configs/models.yaml."""

    name: str
    backend: str
    model_id: str
    family: str | None = None
    kind: str | None = None
    chat_template: str | None = None
    adapter_path: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] | None = None
    request_timeout_s: int | None = None

    @classmethod
    def from_entry(cls, name: str, entry: dict[str, Any], defaults: dict[str, Any]) -> "ModelSpec":
        return cls(
            name=name,
            backend=entry["backend"],
            model_id=entry["model_id"],
            family=entry.get("family"),
            kind=entry.get("kind"),
            chat_template=entry.get("chat_template"),
            adapter_path=entry.get("adapter_path"),
            temperature=entry.get("temperature", defaults.get("temperature")),
            max_tokens=entry.get("max_tokens", defaults.get("max_tokens")),
            extra_body=entry.get("extra_body"),
            request_timeout_s=entry.get("request_timeout_s", defaults.get("request_timeout_s")),
        )


class ModelRegistry:
    """Loads configs/models.yaml and resolves model names to ``ModelSpec``."""

    def __init__(self, path: str | Path = "models.yaml"):
        self._raw = _load_yaml(path)
        self._defaults = self._raw.get("defaults", {})
        self._specs: dict[str, ModelSpec] = {}
        for section in ("targets", "finetunes", "judges"):
            for name, entry in (self._raw.get(section) or {}).items():
                self._specs[name] = ModelSpec.from_entry(name, entry, self._defaults)

    def get(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise KeyError(
                f"Unknown model '{name}'. Known: {sorted(self._specs)}"
            )
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs)

    @property
    def defaults(self) -> dict[str, Any]:
        return dict(self._defaults)


def load_eval_config(path: str | Path = "evaluation.yaml") -> dict[str, Any]:
    return _load_yaml(path)


def load_training_config(path: str | Path = "training.yaml") -> dict[str, Any]:
    return _load_yaml(path)


def output_root() -> Path:
    """Where experiment artefacts are written. Override with $DISTRESS_OUTPUT."""
    root = Path(os.environ.get("DISTRESS_OUTPUT", REPO_ROOT / "outputs"))
    root.mkdir(parents=True, exist_ok=True)
    return root
