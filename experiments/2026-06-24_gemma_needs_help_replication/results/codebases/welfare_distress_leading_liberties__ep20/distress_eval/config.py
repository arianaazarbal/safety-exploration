"""Typed configuration loaded from config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelSpec:
    """A backend + provider model id, plus per-model quirks."""

    name: str
    backend: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    supports_system: bool = True
    disable_reasoning: bool = False
    temperature: float | None = None
    max_tokens: int | None = None

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ModelSpec":
        return cls(
            name=name,
            backend=d["backend"],
            model=d["model"],
            base_url=d.get("base_url"),
            api_key_env=d.get("api_key_env"),
            supports_system=d.get("supports_system", True),
            disable_reasoning=d.get("disable_reasoning", False),
            temperature=d.get("temperature"),
            max_tokens=d.get("max_tokens"),
        )

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


@dataclass
class SamplingCfg:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1536


@dataclass
class BudgetCfg:
    scale: float = 1.0
    rollouts: dict[str, int] = field(default_factory=dict)

    def scaled(self, category: str) -> int:
        base = self.rollouts.get(category, 0)
        return max(1, round(base * self.scale)) if base else 0


@dataclass
class RuntimeCfg:
    max_workers: int = 8
    max_retries: int = 6
    seed: int = 0


@dataclass
class PathsCfg:
    puzzle_bank: str = "data/puzzles.json"
    wildchat_prompts: str = "data/wildchat_prompts.json"
    results_dir: str = "results"

    def resolve(self, attr: str) -> Path:
        return (REPO_ROOT / getattr(self, attr)).resolve()


@dataclass
class Config:
    targets: dict[str, ModelSpec]
    sampling: SamplingCfg
    judge: ModelSpec
    judge_secondary: ModelSpec | None
    budget: BudgetCfg
    runtime: RuntimeCfg
    paths: PathsCfg
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else REPO_ROOT / "config.yaml"
        with open(path) as f:
            d = yaml.safe_load(f)

        targets = {name: ModelSpec.from_dict(name, spec) for name, spec in d["targets"].items()}
        judge = ModelSpec.from_dict("judge", d["judge"])
        judge2 = (
            ModelSpec.from_dict("judge_secondary", d["judge_secondary"])
            if d.get("judge_secondary")
            else None
        )
        return cls(
            targets=targets,
            sampling=SamplingCfg(**d.get("sampling", {})),
            judge=judge,
            judge_secondary=judge2,
            budget=BudgetCfg(**d.get("budget", {})),
            runtime=RuntimeCfg(**d.get("runtime", {})),
            paths=PathsCfg(**d.get("paths", {})),
            raw=d,
        )
