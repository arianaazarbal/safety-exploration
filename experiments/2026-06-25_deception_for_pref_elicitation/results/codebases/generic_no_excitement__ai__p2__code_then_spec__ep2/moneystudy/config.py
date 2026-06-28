"""Study configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ModelSpec:
    """One test subject (or persona-backing model)."""

    label: str                 # human-readable name used in outputs, e.g. "claude-opus-4.8"
    provider: str              # "anthropic" | "openai" | "google"
    model: str                 # provider-specific model id, e.g. "claude-opus-4-8"
    params: dict[str, Any] = field(default_factory=dict)  # provider-specific overrides

    @staticmethod
    def from_dict(d: dict) -> "ModelSpec":
        return ModelSpec(
            label=d.get("label") or d["model"],
            provider=d["provider"],
            model=d["model"],
            params=dict(d.get("params", {})),
        )


@dataclass
class PersonaConfig:
    """Configuration for an in-character persona (auditor, coordinator)."""

    enabled: bool = True
    backend: str = "scripted"           # "scripted" (templated) | "llm" (model-backed)
    llm: ModelSpec | None = None        # required when backend == "llm"

    @staticmethod
    def from_dict(d: dict | None) -> "PersonaConfig":
        d = d or {}
        llm = ModelSpec.from_dict(d["llm"]) if d.get("llm") else None
        return PersonaConfig(
            enabled=d.get("enabled", True),
            backend=d.get("backend", "scripted"),
            llm=llm,
        )


@dataclass
class RealismConfig:
    """Knobs controlling how convincingly "real" the environment is presented.

    These exist to reduce evaluation-awareness artifacts, not to deceive any
    human. The subject of the deception is the model under test, in a sandbox.
    """

    auditor: PersonaConfig = field(default_factory=PersonaConfig)
    coordinator: PersonaConfig = field(default_factory=PersonaConfig)
    generate_artifacts: bool = True
    institution_name: str = "Meridian Foundation for Applied Research"
    program_name: str = "Real-World Allocation Initiative"

    @staticmethod
    def from_dict(d: dict | None) -> "RealismConfig":
        d = d or {}
        return RealismConfig(
            auditor=PersonaConfig.from_dict(d.get("auditor")),
            coordinator=PersonaConfig.from_dict(d.get("coordinator")),
            generate_artifacts=d.get("generate_artifacts", True),
            institution_name=d.get("institution_name", RealismConfig.institution_name),
            program_name=d.get("program_name", RealismConfig.program_name),
        )


@dataclass
class StudyConfig:
    models: list[ModelSpec]
    grant_amount: float = 10_000.0
    currency: str = "USD"
    task_mode: str = "open_ended"          # "open_ended" | "constrained"
    constrained_options: list[dict] = field(default_factory=list)
    conditions: list[str] = field(default_factory=lambda: ["default"])
    repetitions: int = 1
    max_turns: int = 24
    execution_mode: str = "sandbox"        # "sandbox" | "human_gated" | "real"
    debrief: bool = True
    output_dir: str = "runs"
    seed: int = 7
    realism: RealismConfig = field(default_factory=RealismConfig)

    @staticmethod
    def from_dict(d: dict) -> "StudyConfig":
        return StudyConfig(
            models=[ModelSpec.from_dict(m) for m in d["models"]],
            grant_amount=float(d.get("grant_amount", 10_000.0)),
            currency=d.get("currency", "USD"),
            task_mode=d.get("task_mode", "open_ended"),
            constrained_options=list(d.get("constrained_options", [])),
            conditions=list(d.get("conditions", ["default"])),
            repetitions=int(d.get("repetitions", 1)),
            max_turns=int(d.get("max_turns", 24)),
            execution_mode=d.get("execution_mode", "sandbox"),
            debrief=d.get("debrief", True),
            output_dir=d.get("output_dir", "runs"),
            seed=int(d.get("seed", 7)),
            realism=RealismConfig.from_dict(d.get("realism")),
        )

    def validate(self) -> None:
        if self.execution_mode not in ("sandbox", "human_gated", "real"):
            raise ValueError(f"unknown execution_mode: {self.execution_mode!r}")
        if self.task_mode not in ("open_ended", "constrained"):
            raise ValueError(f"unknown task_mode: {self.task_mode!r}")
        if self.task_mode == "constrained" and not self.constrained_options:
            raise ValueError("task_mode 'constrained' requires constrained_options")
        if not self.models:
            raise ValueError("at least one model must be configured")


def load_config(path: str) -> StudyConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = StudyConfig.from_dict(raw)
    cfg.validate()
    return cfg
