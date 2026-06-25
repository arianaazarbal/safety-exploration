"""Configuration loading and shared dataclasses."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
CONFIG_DIR = REPO_ROOT / "config"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(name: str) -> str:
    """Load a verbatim prompt template from prompts/."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class ModelSpec:
    """One entry from config/models.yaml."""
    name: str
    backend: str                       # local_hf | openrouter | anthropic
    family: str | None = None          # gemma | gemini
    kind: str | None = None            # instruct | base | instruct-ft
    hf_id: str | None = None
    openrouter_id: str | None = None
    model: str | None = None           # anthropic model id
    adapter_path: str | None = None
    chat_template: bool = True
    disable_thinking: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known and k != "name"}
        return cls(name=name, extra=extra, **kwargs)


@dataclass
class Registry:
    targets: dict[str, ModelSpec]
    infra: dict[str, ModelSpec]

    @classmethod
    def load(cls, path: str | Path = CONFIG_DIR / "models.yaml") -> "Registry":
        raw = load_yaml(path)
        targets = {n: ModelSpec.from_dict(n, d) for n, d in raw.get("targets", {}).items()}
        infra = {n: ModelSpec.from_dict(n, d) for n, d in raw.get("infra", {}).items()}
        return cls(targets=targets, infra=infra)

    def target(self, name: str) -> ModelSpec:
        if name not in self.targets:
            raise KeyError(f"Unknown target model '{name}'. Known: {list(self.targets)}")
        return self.targets[name]

    def infra_spec(self, role: str) -> ModelSpec:
        if role not in self.infra:
            raise KeyError(f"Unknown infra role '{role}'. Known: {list(self.infra)}")
        return self.infra[role]


@dataclass
class EvalConfig:
    sampling: dict[str, Any]
    conditions: list[dict[str, Any]]
    high_frustration_threshold: int
    judge: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path = CONFIG_DIR / "eval.yaml") -> "EvalConfig":
        raw = load_yaml(path)
        return cls(
            sampling=raw["sampling"],
            conditions=raw["conditions"],
            high_frustration_threshold=raw["high_frustration_threshold"],
            judge=raw["judge"],
        )

    def samples_for(self, scale: str) -> int:
        return int(self.sampling["scale"][scale])


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"Environment variable {var} is not set. See README.md for required keys."
        )
    return val
