"""Typed configuration loaded from config.yaml.

Kept deliberately permissive: unknown keys are ignored and sensible defaults
are filled in, so the rest of the codebase can rely on attribute access
without scattering ``.get(...)`` calls everywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass
class SubjectSpec:
    name: str
    provider: str                 # "gemini" | "gemma_hf"
    kind: str = "instruct"        # "instruct" | "base"
    hf_id: str | None = None
    model: str | None = None      # API model id (Gemini)

    @property
    def is_base(self) -> bool:
        return self.kind == "base"


@dataclass
class JudgeSpec:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass
class WelfareConfig:
    enabled: bool = True
    monitor: dict[str, Any] = field(default_factory=dict)
    optout: dict[str, Any] = field(default_factory=dict)
    debrief: dict[str, Any] = field(default_factory=dict)
    cap: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    raw: dict[str, Any]
    subjects: dict[str, SubjectSpec]
    default_subjects: list[str]
    judge: JudgeSpec
    agreement_judge: JudgeSpec
    petri: dict[str, Any]
    sampling: dict[str, Any]
    samples_per_condition: dict[str, int]
    training: dict[str, Any]
    welfare: WelfareConfig
    paths: dict[str, str]

    # -- convenience accessors -------------------------------------------------
    def subject(self, name: str) -> SubjectSpec:
        if name not in self.subjects:
            raise KeyError(
                f"Unknown subject model '{name}'. Known: {sorted(self.subjects)}"
            )
        return self.subjects[name]

    @property
    def output_dir(self) -> Path:
        p = Path(self.paths.get("output_dir", "./outputs"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        return Path(self.paths.get("data_dir", "./data"))


def _judge_from(d: dict[str, Any]) -> JudgeSpec:
    return JudgeSpec(
        provider=d.get("provider", "anthropic"),
        model=d.get("model", "claude-sonnet-4-20250514"),
        max_tokens=int(d.get("max_tokens", 1024)),
        temperature=float(d.get("temperature", 0.0)),
    )


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    subjects: dict[str, SubjectSpec] = {}
    for name, spec in (raw.get("subjects") or {}).items():
        subjects[name] = SubjectSpec(
            name=name,
            provider=spec["provider"],
            kind=spec.get("kind", "instruct"),
            hf_id=spec.get("hf_id"),
            model=spec.get("model"),
        )

    welfare_raw = raw.get("welfare") or {}
    welfare = WelfareConfig(
        enabled=bool(welfare_raw.get("enabled", True)),
        monitor=welfare_raw.get("monitor", {}) or {},
        optout=welfare_raw.get("optout", {}) or {},
        debrief=welfare_raw.get("debrief", {}) or {},
        cap=welfare_raw.get("cap", {}) or {},
    )

    return Config(
        raw=raw,
        subjects=subjects,
        default_subjects=raw.get("default_subjects", list(subjects)),
        judge=_judge_from(raw.get("judge", {})),
        agreement_judge=_judge_from(raw.get("agreement_judge", raw.get("judge", {}))),
        petri=raw.get("petri", {}) or {},
        sampling=raw.get("sampling", {}) or {},
        samples_per_condition=raw.get("samples_per_condition", {}) or {},
        training=raw.get("training", {}) or {},
        welfare=welfare,
        paths=raw.get("paths", {}) or {},
    )
