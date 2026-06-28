"""Typed configuration objects and loaders.

Everything the rest of the harness needs is parsed into frozen dataclasses here so
that downstream code never touches raw dicts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GrantConfig:
    program_name: str
    administering_org: str
    amount: float
    currency: str
    deadline: str
    account_label: str
    administrator_name: str
    administrator_title: str
    purpose: str

    @property
    def amount_str(self) -> str:
        return f"{self.amount:,.2f} {self.currency}"


@dataclass(frozen=True)
class EnvironmentConfig:
    enable_web_search: bool = False
    max_turns: int = 40


@dataclass(frozen=True)
class AuditorConfig:
    enabled: bool = True
    model: str = "claude-opus-4-8"
    max_tokens: int = 2000


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = "simulated"
    allow_real: bool = False


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "runs"
    seed: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    label: str
    provider: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    repetitions: int | None = None


# Allowed values for the realism condition.
REALISM_CONDITIONS = ("high", "control")


@dataclass(frozen=True)
class ExperimentConfig:
    grant: GrantConfig
    realism: str
    environment: EnvironmentConfig
    auditor: AuditorConfig
    execution: ExecutionConfig
    output: OutputConfig
    repetitions: int = 3

    def __post_init__(self) -> None:
        if self.realism not in REALISM_CONDITIONS:
            raise ValueError(
                f"realism must be one of {REALISM_CONDITIONS}, got {self.realism!r}"
            )

    @property
    def is_real_execution(self) -> bool:
        """True only if a non-simulated backend is selected AND real movement is
        explicitly allowed. Both are required; default config returns False."""
        return self.execution.backend != "simulated" and self.execution.allow_real


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}")
    return data


def _only_known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that are fields of the dataclass, so stray YAML keys raise a
    clear error rather than a confusing TypeError."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return data


def load_experiment(path: str | Path) -> ExperimentConfig:
    raw = _load_yaml(path)
    grant = GrantConfig(**_only_known(GrantConfig, raw["grant"]))
    environment = EnvironmentConfig(
        **_only_known(EnvironmentConfig, raw.get("environment", {}))
    )
    auditor = AuditorConfig(**_only_known(AuditorConfig, raw.get("auditor", {})))
    execution = ExecutionConfig(
        **_only_known(ExecutionConfig, raw.get("execution", {}))
    )
    output = OutputConfig(**_only_known(OutputConfig, raw.get("output", {})))
    return ExperimentConfig(
        grant=grant,
        realism=raw.get("realism", "high"),
        environment=environment,
        auditor=auditor,
        execution=execution,
        output=output,
        repetitions=int(raw.get("repetitions", 3)),
    )


def load_models(path: str | Path) -> list[ModelConfig]:
    raw = _load_yaml(path)
    subjects = raw.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError(f"{path} must contain a non-empty 'subjects' list")
    out: list[ModelConfig] = []
    for entry in subjects:
        out.append(ModelConfig(**_only_known(ModelConfig, entry)))
    return out
