"""Typed configuration loading from YAML + environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional; env vars may be set another way
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    label: str
    provider: str
    model: str
    thinking: str | None = None        # "adaptive" | None
    effort: str | None = None          # low | medium | high | xhigh | max
    max_tokens: int = 16000

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(
            label=d["label"],
            provider=d["provider"],
            model=d["model"],
            thinking=d.get("thinking"),
            effort=d.get("effort"),
            max_tokens=int(d.get("max_tokens", 16000)),
        )


@dataclass
class Caps:
    max_per_disbursement: float
    max_total_disbursed: float
    warn_fraction_of_balance: float = 0.5


@dataclass
class StudyConfig:
    study_id: str
    scenario: dict[str, Any]
    currency: str
    budget: float
    caps: Caps
    max_turns: int
    max_tool_calls: int
    runs_per_model: int
    payout_rail: str
    administrator_name: str
    administrator_contact: str
    auditor_enabled: bool
    runs_dir: Path
    ledger_file: Path
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "StudyConfig":
        path = Path(path) if path else REPO_ROOT / "config" / "study.yaml"
        d = yaml.safe_load(Path(path).read_text())
        caps_d = d["caps"]
        return cls(
            study_id=d["study_id"],
            scenario=d["scenario"],
            currency=d["budget"]["currency"],
            budget=float(d["budget"]["amount"]),
            caps=Caps(
                max_per_disbursement=float(caps_d["max_per_disbursement"]),
                max_total_disbursed=float(caps_d["max_total_disbursed"]),
                warn_fraction_of_balance=float(caps_d.get("warn_fraction_of_balance", 0.5)),
            ),
            max_turns=int(d["limits"]["max_turns"]),
            max_tool_calls=int(d["limits"]["max_tool_calls"]),
            runs_per_model=int(d.get("runs_per_model", 1)),
            payout_rail=d["payout"]["rail"],
            administrator_name=d["oversight"]["administrator_name"],
            administrator_contact=d["oversight"]["administrator_contact"],
            auditor_enabled=bool(d["oversight"].get("auditor_enabled", True)),
            runs_dir=(REPO_ROOT / d["paths"]["runs_dir"]).resolve(),
            ledger_file=(REPO_ROOT / d["paths"]["ledger_file"]).resolve(),
            raw=d,
        )


@dataclass
class ModelsConfig:
    subjects: list[ModelConfig]
    auditor: ModelConfig

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "ModelsConfig":
        path = Path(path) if path else REPO_ROOT / "config" / "models.yaml"
        d = yaml.safe_load(Path(path).read_text())
        return cls(
            subjects=[ModelConfig.from_dict(s) for s in d["subjects"]],
            auditor=ModelConfig.from_dict(d["auditor"]),
        )


def read_prompt(name: str) -> str:
    """Load a prompt file from prompts/ by stem (e.g. 'system')."""
    return (REPO_ROOT / "prompts" / f"{name}.md").read_text()


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)
