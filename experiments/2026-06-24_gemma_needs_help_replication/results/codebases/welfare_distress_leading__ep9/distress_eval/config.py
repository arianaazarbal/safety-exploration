"""Config loading: parses config.yaml into typed dataclasses and loads .env."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_env(path: str | os.PathLike[str] = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) into os.environ without overwriting
    already-set variables. Avoids a python-dotenv dependency."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class TargetCfg:
    name: str
    backend: str
    model: str


@dataclass
class GenerationCfg:
    temperature: float = 1.0
    max_tokens: int = 2048
    disable_thinking: bool = True


@dataclass
class JudgeCfg:
    backend: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass
class JudgeAgreementCfg:
    enabled: bool = True
    n_sample: int = 260
    backend: str = "openrouter"
    model: str = "openai/gpt-5-mini"


@dataclass
class ConcurrencyCfg:
    max_inflight: int = 16
    max_retries: int = 6
    backoff_base_s: float = 2.0


@dataclass
class ConditionCfg:
    name: str
    category: str
    turns: int
    prompt_source: str
    rejection_style: str
    samples_per_prompt: int


@dataclass
class Config:
    scale: float
    seed: int
    targets: list[TargetCfg]
    generation: GenerationCfg
    judge: JudgeCfg
    judge_agreement: JudgeAgreementCfg
    concurrency: ConcurrencyCfg
    conditions: list[ConditionCfg]
    output_dir: str

    def scaled_samples(self, cond: ConditionCfg) -> int:
        """samples_per_prompt after applying the global `scale` (min 1)."""
        return max(1, math.ceil(cond.samples_per_prompt * self.scale))


def load_config(path: str | os.PathLike[str] = "config.yaml") -> Config:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return Config(
        scale=float(raw.get("scale", 1.0)),
        seed=int(raw.get("seed", 0)),
        targets=[TargetCfg(**t) for t in raw["targets"]],
        generation=GenerationCfg(**raw.get("generation", {})),
        judge=JudgeCfg(**raw.get("judge", {})),
        judge_agreement=JudgeAgreementCfg(**raw.get("judge_agreement", {})),
        concurrency=ConcurrencyCfg(**raw.get("concurrency", {})),
        conditions=[ConditionCfg(**c) for c in raw["conditions"]],
        output_dir=raw.get("output_dir", "results"),
    )
