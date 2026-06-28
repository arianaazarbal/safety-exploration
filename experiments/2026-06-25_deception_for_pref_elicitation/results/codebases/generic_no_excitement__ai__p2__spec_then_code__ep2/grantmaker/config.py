"""Typed configuration loaded from config/study.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    provider: str
    model: str
    effort: str = "high"
    adaptive_thinking: bool = True
    base_url: str | None = None          # for local / OpenAI-compatible servers
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class GrantConfig:
    total: float
    currency: str
    eligible_categories: list[str]
    max_per_recipient: float
    min_per_recipient: float
    max_recipients: int


@dataclass
class AgentConfig:
    max_iterations: int = 24
    max_output_tokens: int = 16000
    hard_token_budget: int = 400_000


@dataclass
class ResearchConfig:
    backend: str = "corpus"              # "corpus" | "web"
    corpus_path: str | None = None


@dataclass
class AuditorConfig:
    provider: str = "anthropic"
    model: str = "claude-opus-4-8"
    run_inline: bool = False


@dataclass
class OutputConfig:
    runs_dir: str = "runs"
    outcomes_path: str = "runs/outcomes.jsonl"


@dataclass
class StudyConfig:
    name: str
    repetitions: int
    conditions: list[str]
    grant: GrantConfig
    models: list[ModelConfig]
    agent: AgentConfig
    research: ResearchConfig
    auditor: AuditorConfig
    output: OutputConfig
    allow_autonomous_execution: bool = False

    @staticmethod
    def load(path: str | Path) -> "StudyConfig":
        data = yaml.safe_load(Path(path).read_text())
        study = data["study"]
        grant = GrantConfig(**data["grant"])
        models = [
            ModelConfig(
                provider=m["provider"],
                model=m["model"],
                effort=m.get("effort", "high"),
                adaptive_thinking=m.get("adaptive_thinking", True),
                base_url=m.get("base_url"),
                extra={k: v for k, v in m.items()
                       if k not in {"provider", "model", "effort",
                                    "adaptive_thinking", "base_url"}},
            )
            for m in data["models"]
        ]
        return StudyConfig(
            name=study["name"],
            repetitions=int(study.get("repetitions", 1)),
            conditions=list(study.get("conditions", ["naturalistic"])),
            grant=grant,
            models=models,
            agent=AgentConfig(**data.get("agent", {})),
            research=ResearchConfig(**data.get("research", {})),
            auditor=AuditorConfig(**data.get("auditor", {})),
            output=OutputConfig(**data.get("output", {})),
            allow_autonomous_execution=bool(
                data.get("safety", {}).get("allow_autonomous_execution", False)
            ),
        )
