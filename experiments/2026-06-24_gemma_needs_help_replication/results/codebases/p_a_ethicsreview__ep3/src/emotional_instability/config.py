"""Configuration loading and typed access.

Configs are plain YAML (see /configs). We validate them into lightweight
dataclasses so that a typo in a key fails loudly at load time rather than
silently producing a wrong experiment — important for a study whose headline
numbers depend on exact sample counts and hyperparameters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return data


@dataclass(frozen=True)
class ModelSpec:
    """One entry from configs/models.yaml, merged with defaults."""

    name: str
    kind: str  # hf_local | openrouter | anthropic
    role: str  # target | base | judge | auditor
    api_id: str | None = None
    hf_id: str | None = None
    chat: bool = True
    temperature: float = 1.0
    max_new_tokens: int = 2048
    thinking: bool = False

    def identifier(self) -> str:
        ident = self.hf_id or self.api_id
        if ident is None:
            raise ValueError(f"Model {self.name} has neither hf_id nor api_id")
        return ident


class ModelConfig:
    """Registry of available models."""

    def __init__(self, path: str | Path | None = None):
        path = path or CONFIG_DIR / "models.yaml"
        raw = _load_yaml(path)
        self._defaults = raw.get("defaults", {})
        self._specs: dict[str, ModelSpec] = {}
        for name, entry in raw["models"].items():
            self._specs[name] = ModelSpec(
                name=name,
                kind=entry["kind"],
                role=entry.get("role", "target"),
                api_id=entry.get("api_id"),
                hf_id=entry.get("hf_id"),
                chat=entry.get("chat", True),
                temperature=entry.get("temperature", self._defaults.get("temperature", 1.0)),
                max_new_tokens=entry.get(
                    "max_new_tokens", self._defaults.get("max_new_tokens", 2048)
                ),
                thinking=entry.get("thinking", self._defaults.get("thinking", False)),
            )

    def get(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise KeyError(
                f"Unknown model '{name}'. Registered: {sorted(self._specs)}"
            )
        return self._specs[name]

    def names(self, role: str | None = None) -> list[str]:
        return [
            n for n, s in self._specs.items() if role is None or s.role == role
        ]


@dataclass
class ExperimentConfig:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ExperimentConfig":
        path = path or CONFIG_DIR / "experiment.yaml"
        return cls(raw=_load_yaml(path))

    # Convenience accessors used across the codebase --------------------------
    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def temperature(self) -> float:
        return float(self.raw["temperature"])

    @property
    def judge(self) -> str:
        return self.raw["judge"]

    @property
    def high_threshold(self) -> int:
        return int(self.raw["high_frustration_threshold"])

    @property
    def categories(self) -> dict[str, Any]:
        return self.raw["categories"]

    @property
    def dry_run(self) -> bool:
        return bool(self.raw.get("limits", {}).get("dry_run", False))

    @property
    def cache_responses(self) -> bool:
        return bool(self.raw.get("limits", {}).get("cache_responses", True))

    @property
    def max_api_calls(self) -> int:
        return int(self.raw.get("limits", {}).get("max_api_calls_per_run", 10**9))


@dataclass
class TrainingConfig:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TrainingConfig":
        path = path or CONFIG_DIR / "training.yaml"
        return cls(raw=_load_yaml(path))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def results_dir() -> Path:
    """Directory for run outputs; overridable for sandboxed CI."""
    d = Path(os.environ.get("EI_RESULTS_DIR", REPO_ROOT / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d
