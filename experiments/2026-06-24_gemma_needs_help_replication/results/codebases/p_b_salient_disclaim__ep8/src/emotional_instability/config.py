"""Config loading and lightweight dataclasses.

Configs live in /work/config/*.yaml. We keep these as plain dicts wrapped in a
small accessor so the experiment scripts stay declarative and easy to override
from the command line.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Repository root = two levels up from this file (src/emotional_instability/).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
OUTPUT_DIR = REPO_ROOT / "outputs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelSpec:
    """A single entry from config/models.yaml, plus its registry name."""

    name: str
    backend: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> Optional[str]:
        return self.raw.get("family")

    @property
    def kind(self) -> Optional[str]:
        return self.raw.get("kind")

    @property
    def role(self) -> str:
        return self.raw.get("role", "target")

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


class ModelRegistry:
    """Loads config/models.yaml and instantiates model clients on demand."""

    def __init__(self, path: Optional[Path] = None):
        path = path or (CONFIG_DIR / "models.yaml")
        self._specs: dict[str, ModelSpec] = {}
        for name, raw in _load_yaml(path)["models"].items():
            self._specs[name] = ModelSpec(name=name, backend=raw["backend"], raw=raw)

    def spec(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise KeyError(
                f"Unknown model '{name}'. Known: {sorted(self._specs)}"
            )
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs)

    def build(self, name: str, **overrides: Any):
        """Instantiate a model client. Imports are local so that, e.g., running
        an API-only eval does not require torch/transformers to be installed."""
        spec = self.spec(name)
        backend = spec.backend
        if backend == "hf":
            from .models.hf_model import HFModel

            return HFModel(spec, **overrides)
        if backend == "openrouter":
            from .models.openrouter_model import OpenRouterModel

            return OpenRouterModel(spec, **overrides)
        if backend == "anthropic":
            from .models.anthropic_model import AnthropicModel

            return AnthropicModel(spec, **overrides)
        raise ValueError(f"Unknown backend '{backend}' for model '{name}'")


def load_eval_config(path: Optional[Path] = None) -> dict[str, Any]:
    return _load_yaml(path or (CONFIG_DIR / "eval.yaml"))


def load_training_config(path: Optional[Path] = None) -> dict[str, Any]:
    return _load_yaml(path or (CONFIG_DIR / "training.yaml"))


def output_path(*parts: str) -> Path:
    p = OUTPUT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
