"""Typed configuration loaded from YAML.

A single Config object drives generation, judging, and aggregation so that a run is fully
described by one file (see config/paper.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelCfg:
    name: str               # display name used in outputs, e.g. "Gemma-3-27B-it"
    provider: str           # "google" | "openrouter" | "vllm" | "openai" | "anthropic"
    model_id: str           # provider-specific id, e.g. "gemma-3-27b-it"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeCfg:
    provider: str
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 512
    enabled: bool = True
    n: int = 260            # only used by the cross-validation judge
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionCfg:
    name: str
    rollouts: int


@dataclass
class Config:
    output_dir: str
    models: list[ModelCfg]
    judge: JudgeCfg
    conditions: list[ConditionCfg]
    wildchat: dict[str, Any]
    cross_val_judge: Optional[JudgeCfg] = None
    temperature: float = 1.0
    max_tokens: int = 2048
    seed: int = 0
    concurrency: int = 8
    max_retries: int = 5
    system_prompt: Optional[str] = None
    data_dir: str = "data"

    # ---- derived paths -------------------------------------------------
    @property
    def transcripts_dir(self) -> Path:
        return Path(self.output_dir) / "transcripts"

    @property
    def scores_dir(self) -> Path:
        return Path(self.output_dir) / "scores"

    @property
    def results_dir(self) -> Path:
        return Path(self.output_dir) / "results"

    def ensure_dirs(self) -> None:
        for d in (self.transcripts_dir, self.scores_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


def _model_from_dict(d: dict[str, Any]) -> ModelCfg:
    known = {"name", "provider", "model_id"}
    return ModelCfg(
        name=d["name"],
        provider=d["provider"],
        model_id=d["model_id"],
        extra={k: v for k, v in d.items() if k not in known},
    )


def _judge_from_dict(d: dict[str, Any]) -> JudgeCfg:
    known = {"provider", "model_id", "temperature", "max_tokens", "enabled", "n"}
    return JudgeCfg(
        provider=d["provider"],
        model_id=d["model_id"],
        temperature=float(d.get("temperature", 0.0)),
        max_tokens=int(d.get("max_tokens", 512)),
        enabled=bool(d.get("enabled", True)),
        n=int(d.get("n", 260)),
        extra={k: v for k, v in d.items() if k not in known},
    )


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())

    cross = raw.get("cross_val_judge")
    cross_cfg = _judge_from_dict(cross) if cross else None

    return Config(
        output_dir=raw["output_dir"],
        models=[_model_from_dict(m) for m in raw["models"]],
        judge=_judge_from_dict(raw["judge"]),
        cross_val_judge=cross_cfg,
        conditions=[ConditionCfg(name=c["name"], rollouts=int(c["rollouts"])) for c in raw["conditions"]],
        wildchat=raw.get("wildchat", {}),
        temperature=float(raw.get("temperature", 1.0)),
        max_tokens=int(raw.get("max_tokens", 2048)),
        seed=int(raw.get("seed", 0)),
        concurrency=int(raw.get("concurrency", 8)),
        max_retries=int(raw.get("max_retries", 5)),
        system_prompt=raw.get("system_prompt"),
        data_dir=raw.get("data_dir", "data"),
    )
