"""Configuration loading.

Configs live as YAML under ``config/`` and are loaded into light dataclasses.
We deliberately avoid a heavy schema library; the dataclasses document the
expected shape and provide attribute access while still allowing extra keys via
the raw dict (kept on ``.raw``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/distress/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
OUTPUTS_DIR = Path(os.environ.get("DISTRESS_OUTPUTS", REPO_ROOT / "outputs"))
DATA_DIR = REPO_ROOT / "data"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    with open(path) as fh:
        return yaml.safe_load(fh)


@dataclass
class ModelSpec:
    """One entry from models.yaml (target, finetuned, or role)."""

    name: str
    backend: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def hf_id(self) -> str | None:
        return self.raw.get("hf_id")

    @property
    def api_id(self) -> str | None:
        return self.raw.get("api_id")

    @property
    def kind(self) -> str:
        return self.raw.get("kind", "instruct")

    @property
    def family(self) -> str:
        return self.raw.get("family", "")

    @property
    def is_chat(self) -> bool:
        return bool(self.raw.get("chat", self.kind == "instruct"))

    @property
    def adapter_path(self) -> str | None:
        return self.raw.get("adapter_path")

    @property
    def base(self) -> str | None:
        return self.raw.get("base")


class ModelRegistry:
    """Loads models.yaml and resolves names to :class:`ModelSpec`."""

    def __init__(self, path: str | Path = "models.yaml"):
        self.raw = load_yaml(path)
        self._specs: dict[str, ModelSpec] = {}
        for section in ("targets", "finetuned", "roles"):
            for name, entry in (self.raw.get(section) or {}).items():
                self._specs[name] = ModelSpec(name=name, backend=entry["backend"], raw=entry)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def get(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self._specs)}")
        return self._specs[name]

    def role(self, role_name: str) -> ModelSpec:
        entry = (self.raw.get("roles") or {}).get(role_name)
        if entry is None:
            raise KeyError(f"Unknown role '{role_name}'")
        return ModelSpec(name=role_name, backend=entry["backend"], raw=entry)

    @property
    def target_names(self) -> list[str]:
        return list((self.raw.get("targets") or {}).keys())


def load_experiment(path: str | Path = "experiment.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_training(path: str | Path = "training.yaml") -> dict[str, Any]:
    return load_yaml(path)
