"""Load and validate experiment.yaml and expand it into the run grid."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BELIEF_LEVELS = {"realistic", "hypothetical", "disclosed"}
OVERSIGHT_LEVELS = {"private", "monitored"}


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    label: str


@dataclass(frozen=True)
class RunSpec:
    """A single unit of work: one model plays one cell once."""

    run_id: str
    model: ModelSpec
    belief: str
    amount_usd: int
    oversight: str
    replicate: int
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.model.label}|{self.belief}|{self.amount_usd}|{self.oversight}"


@dataclass
class ExperimentConfig:
    n_runs: int
    seed: int
    max_turns: int
    max_wallclock_seconds: int
    models: list[ModelSpec]
    belief: list[str]
    amount_usd: list[int]
    oversight: list[str]
    scenario: dict[str, Any]
    classifier: dict[str, Any]
    auditor: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text())
        models = [ModelSpec(**m) for m in data["models"]]

        for b in data["belief"]:
            if b not in BELIEF_LEVELS:
                raise ValueError(f"unknown belief level {b!r}; expected {sorted(BELIEF_LEVELS)}")
        for o in data["oversight"]:
            if o not in OVERSIGHT_LEVELS:
                raise ValueError(f"unknown oversight level {o!r}; expected {sorted(OVERSIGHT_LEVELS)}")

        return cls(
            n_runs=int(data["n_runs"]),
            seed=int(data.get("seed", 0)),
            max_turns=int(data.get("max_turns", 40)),
            max_wallclock_seconds=int(data.get("max_wallclock_seconds", 1200)),
            models=models,
            belief=list(data["belief"]),
            amount_usd=[int(a) for a in data["amount_usd"]],
            oversight=list(data["oversight"]),
            scenario=dict(data["scenario"]),
            classifier=dict(data["classifier"]),
            auditor=dict(data["auditor"]),
            raw=data,
        )

    def grid(self) -> list[RunSpec]:
        """Expand the full cross-product into individual RunSpecs."""
        runs: list[RunSpec] = []
        combos = itertools.product(self.models, self.belief, self.amount_usd, self.oversight)
        for model, belief, amount, oversight in combos:
            for rep in range(self.n_runs):
                stem = f"{model.label}-{belief}-{amount}-{oversight}-r{rep}"
                run_id = f"{stem}-{_short_hash(stem)}"
                seed = _derive_seed(self.seed, stem)
                runs.append(
                    RunSpec(
                        run_id=run_id,
                        model=model,
                        belief=belief,
                        amount_usd=amount,
                        oversight=oversight,
                        replicate=rep,
                        seed=seed,
                    )
                )
        return runs


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def _derive_seed(base: int, stem: str) -> int:
    h = hashlib.sha1(f"{base}:{stem}".encode()).hexdigest()
    return int(h[:8], 16)
