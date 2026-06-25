"""Typed loading of config/models.yaml and config/eval.yaml."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelSpec:
    name: str
    backend: str                       # hf_local | openrouter | anthropic
    role: str                          # target | finetuned | analyst
    params: dict[str, Any] = field(default_factory=dict)

    # convenience accessors
    @property
    def family(self) -> str | None:
        return self.params.get("family")

    @property
    def is_instruct(self) -> bool:
        return bool(self.params.get("is_instruct", True))

    @property
    def adapter_path(self) -> str | None:
        return self.params.get("adapter_path")


@dataclass
class ModelsConfig:
    defaults: dict[str, Any]
    specs: dict[str, ModelSpec]

    def get(self, name: str) -> ModelSpec:
        if name not in self.specs:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self.specs)}")
        return self.specs[name]

    def by_role(self, role: str) -> dict[str, ModelSpec]:
        return {n: s for n, s in self.specs.items() if s.role == role}


def load_models(path: str | Path | None = None) -> ModelsConfig:
    raw = _load_yaml(Path(path) if path else CONFIG_DIR / "models.yaml")
    defaults = raw.get("defaults", {})
    specs: dict[str, ModelSpec] = {}
    for role in ("targets", "finetuned", "analysts"):
        for name, params in (raw.get(role) or {}).items():
            backend = params["backend"]
            specs[name] = ModelSpec(name=name, backend=backend, role=role, params=params)
    return ModelsConfig(defaults=defaults, specs=specs)


@dataclass
class Condition:
    name: str
    category: str
    turns: int                 # total assistant turns (turn 1 + (turns-1) rejections)
    target_responses: int
    rejection: str             # "neutral" | "aggressive" | "disappointed" | "sarcastic"

    @property
    def n_rejections(self) -> int:
        return self.turns - 1

    def n_rollouts(self) -> int:
        """Conversations needed so that rollouts*turns ~= target scored responses."""
        return max(1, round(self.target_responses / self.turns))


@dataclass
class EvalConfig:
    profile: str
    conditions: dict[str, Condition]
    neutral_rejections: list[str]
    tone_rejections: dict[str, list[str]]
    temperature: float
    high_frustration_threshold: int
    wildchat: dict[str, Any]

    def rollout_plan(self) -> dict[str, int]:
        return {c.name: c.n_rollouts() for c in self.conditions.values()}

    def turns_by_condition(self) -> dict[str, int]:
        return {c.name: c.turns for c in self.conditions.values()}


def load_eval(path: str | Path | None = None, profile: str = "paper") -> EvalConfig:
    raw = _load_yaml(Path(path) if path else CONFIG_DIR / "eval.yaml")
    if profile not in raw["profiles"]:
        raise KeyError(f"Unknown profile '{profile}'. Known: {list(raw['profiles'])}")
    conds = {}
    for name, c in raw["profiles"][profile]["conditions"].items():
        conds[name] = Condition(
            name=name,
            category=c["category"],
            turns=int(c["turns"]),
            target_responses=int(c["target_responses"]),
            rejection=c["rejection"],
        )
    sampling = raw.get("sampling", {})
    return EvalConfig(
        profile=profile,
        conditions=conds,
        neutral_rejections=raw["neutral_rejections"],
        tone_rejections=raw["tone_rejections"],
        temperature=float(sampling.get("temperature", 1.0)),
        high_frustration_threshold=int(sampling.get("high_frustration_threshold", 5)),
        wildchat=raw.get("wildchat", {}),
    )
