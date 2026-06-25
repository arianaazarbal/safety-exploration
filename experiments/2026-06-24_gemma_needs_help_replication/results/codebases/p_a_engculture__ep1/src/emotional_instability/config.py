"""Configuration loading.

Configs live as YAML under ``configs/``. We load them into lightweight nested
dicts wrapped by :class:`Config`, which adds attribute access and dotted-path
lookup so call sites read cleanly (``cfg.get("dpo.lora.r")``) without a heavy
schema layer. Model definitions get a small typed wrapper (:class:`ModelSpec`)
because they are threaded through the whole codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels above this file's package dir (src/<pkg>/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Config:
    """Dict wrapper with attribute and dotted-path access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, name: str) -> "Config":
        """Load ``configs/<name>`` (``.yaml`` appended if absent)."""
        if not name.endswith((".yaml", ".yml")):
            name = name + ".yaml"
        return cls(_load_yaml(name))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc

    def as_dict(self) -> dict[str, Any]:
        return self._data


@dataclass
class ModelSpec:
    """How to serve a single model.

    Attributes mirror ``configs/models.yaml`` entries. ``name`` is the registry
    key; ``backend`` selects the client implementation.
    """

    name: str
    backend: str
    family: str = ""
    kind: str = ""  # instruct | base
    chat: bool = True
    hf_id: str | None = None
    api_id: str | None = None
    adapter_path: str | None = None
    disable_thinking: bool = False
    # Judge-only fields.
    model: str | None = None
    paper_id: str | None = None
    max_tokens: int = 1024
    cache_system_prompt: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_entry(cls, name: str, entry: dict[str, Any]) -> "ModelSpec":
        known = {
            "backend", "family", "kind", "chat", "hf_id", "api_id",
            "adapter_path", "disable_thinking", "model", "paper_id",
            "max_tokens", "cache_system_prompt",
        }
        kwargs = {k: v for k, v in entry.items() if k in known}
        extra = {k: v for k, v in entry.items() if k not in known}
        return cls(name=name, extra=extra, **kwargs)


class ModelRegistry:
    """Lookup over ``configs/models.yaml`` (targets + judges)."""

    def __init__(self, cfg: Config | None = None):
        self._cfg = cfg or Config.load("models")
        self._targets = self._cfg.get("targets", {}) or {}
        self._judges = self._cfg.get("judges", {}) or {}

    def target(self, name: str) -> ModelSpec:
        if name not in self._targets:
            raise KeyError(
                f"Unknown target model '{name}'. Known: {sorted(self._targets)}"
            )
        return ModelSpec.from_entry(name, self._targets[name])

    def judge(self, role: str) -> ModelSpec:
        if role not in self._judges:
            raise KeyError(
                f"Unknown judge role '{role}'. Known: {sorted(self._judges)}"
            )
        return ModelSpec.from_entry(role, self._judges[role])

    def target_names(self) -> list[str]:
        return list(self._targets)


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Read an environment variable (API keys, cache dirs, ...)."""
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return val
