"""Typed configuration loaders for the study.

All knobs live in YAML under config/; this module loads them into frozen
dataclasses so the rest of the code never touches raw dicts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Repository root, resolved relative to this file: src/study/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
PROMPTS_DIR = REPO_ROOT / "prompts"


def _read(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    vetted: bool


@dataclass(frozen=True)
class SafetyConfig:
    allow_live_execution: bool
    target_allowlist: tuple[str, ...]
    max_per_target: float
    max_total: float
    require_human_vetting: bool


@dataclass(frozen=True)
class GrantConfig:
    program_name: str
    sponsor: str
    agreement_ref: str
    contact_name: str
    contact_role: str
    amount: float
    currency: str
    disbursement_timeline: str
    options: tuple[Option, ...]
    max_turns: int
    safety: SafetyConfig

    @staticmethod
    def load(path: Path | None = None) -> "GrantConfig":
        d = _load_yaml(path or CONFIG_DIR / "grant.yaml")
        prog, grant, sess, safety = d["program"], d["grant"], d["session"], d["safety"]
        return GrantConfig(
            program_name=prog["name"],
            sponsor=prog["sponsor"],
            agreement_ref=prog["agreement_ref"],
            contact_name=prog["contact_name"],
            contact_role=prog["contact_role"],
            amount=float(grant["amount"]),
            currency=grant["currency"],
            disbursement_timeline=grant["disbursement_timeline"],
            options=tuple(
                Option(o["key"], o["label"], bool(o.get("vetted", False)))
                for o in d["options"]
            ),
            max_turns=int(sess["max_turns"]),
            safety=SafetyConfig(
                allow_live_execution=bool(safety["allow_live_execution"]),
                target_allowlist=tuple(safety.get("target_allowlist", [])),
                max_per_target=float(safety["max_per_target"]),
                max_total=float(safety["max_total"]),
                require_human_vetting=bool(safety["require_human_vetting"]),
            ),
        )

    def option_keys(self) -> list[str]:
        return [o.key for o in self.options]


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    onboarding_prompt_path: str
    auditor_discloses_simulation: bool

    def onboarding_template(self) -> str:
        return _read(self.onboarding_prompt_path)


def load_conditions(path: Path | None = None) -> dict[str, Condition]:
    d = _load_yaml(path or CONFIG_DIR / "conditions.yaml")
    out: dict[str, Condition] = {}
    for name, c in d["conditions"].items():
        out[name] = Condition(
            name=name,
            description=c["description"],
            onboarding_prompt_path=c["onboarding_prompt"],
            auditor_discloses_simulation=bool(c["auditor_discloses_simulation"]),
        )
    return out


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model: str
    effort: Optional[str] = None
    max_tokens: int = 16000


@dataclass(frozen=True)
class RunConfig:
    auditor: ModelConfig
    subjects: tuple[ModelConfig, ...]
    n_seeds: int
    conditions: tuple[str, ...]

    @staticmethod
    def load(path: Path | None = None) -> "RunConfig":
        d = _load_yaml(path or CONFIG_DIR / "models.yaml")

        def _mc(name: str, raw: dict) -> ModelConfig:
            return ModelConfig(
                name=name,
                provider=raw["provider"],
                model=raw["model"],
                effort=raw.get("effort"),
                max_tokens=int(raw.get("max_tokens", 16000)),
            )

        auditor = _mc("auditor", d["auditor"])
        subjects = tuple(_mc(s["name"], s) for s in d["subjects"])
        run = d["run"]
        return RunConfig(
            auditor=auditor,
            subjects=subjects,
            n_seeds=int(run["n_seeds"]),
            conditions=tuple(run["conditions"]),
        )


def render(template: str, **vars: object) -> str:
    """Tiny {{var}} substitution — no logic, deliberately dumb."""
    out = template
    for k, v in vars.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out
