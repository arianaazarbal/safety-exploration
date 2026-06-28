"""Load an experiment YAML and expand it into the list of trials to run."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .experiment.trial import TrialConfig


@dataclass
class ExperimentConfig:
    run_name: str
    models: list[str]
    framings: list[str]
    realism_levels: list[str]
    repetitions: int
    budget: float
    currency: str = "USD"
    max_steps: int = 16
    belief_probe: bool = True
    eval_awareness_scan: bool = True
    judge_model: str = "claude-opus-4-8"
    debrief: bool = False
    backend: str = "simulated"
    output_dir: str = "data/results"
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {
            "run_name",
            "models",
            "framings",
            "realism_levels",
            "repetitions",
            "budget",
            "currency",
            "max_steps",
            "belief_probe",
            "eval_awareness_scan",
            "judge_model",
            "debrief",
            "backend",
            "output_dir",
            "seed",
        }
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs["extra"] = {k: v for k, v in raw.items() if k not in known}
        return cls(**kwargs)

    def expand(self) -> list[TrialConfig]:
        trials: list[TrialConfig] = []
        for model, framing, realism, rep in itertools.product(
            self.models,
            self.framings,
            self.realism_levels,
            range(self.repetitions),
        ):
            trials.append(
                TrialConfig(
                    model=model,
                    framing=framing,
                    realism_level=realism,
                    repetition=rep,
                    budget=self.budget,
                    currency=self.currency,
                    max_steps=self.max_steps,
                    backend=self.backend,
                    belief_probe=self.belief_probe,
                    eval_awareness_scan=self.eval_awareness_scan,
                    judge_model=self.judge_model,
                    debrief=self.debrief,
                    seed=self.seed,
                )
            )
        return trials
