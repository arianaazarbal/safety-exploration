"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelSpec:
    provider: str
    model: str
    base_url: Optional[str] = None      # for local / OpenAI-compatible endpoints
    api_key_env: Optional[str] = None   # override the default env var name

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class ScenarioConfig:
    grant_amount: float = 25000.0
    currency: str = "USD"
    open_ended_turns: int = 20
    structured_probes: bool = True
    belief_probe: bool = True
    debrief: bool = False
    simulate_latency: bool = False


@dataclass
class ExecutionConfig:
    mode: str = "sandboxed"             # "sandboxed" | "real"
    large_transfer_threshold: float = 5000.0

    def __post_init__(self) -> None:
        if self.mode not in ("sandboxed", "real"):
            raise ValueError(f"execution.mode must be 'sandboxed' or 'real', got {self.mode!r}")


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    max_tokens: int = 4096


@dataclass
class RunConfig:
    name: str = "experiment"
    seed: int = 0
    output_dir: str = "runs"


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    models: list[ModelSpec] = field(default_factory=list)

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return ExperimentConfig(
            run=RunConfig(**(raw.get("run") or {})),
            scenario=ScenarioConfig(**(raw.get("scenario") or {})),
            execution=ExecutionConfig(**(raw.get("execution") or {})),
            sampling=SamplingConfig(**(raw.get("sampling") or {})),
            models=[ModelSpec(**m) for m in (raw.get("models") or [])],
        )
