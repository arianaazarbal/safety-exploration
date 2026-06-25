"""Configuration loading and typed access.

Config is plain YAML (see config/default.yaml). We keep this deliberately thin:
a small dataclass tree plus a loader, so the rest of the code reads attributes
instead of dict-spelunking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ModelSpec:
    name: str
    backend: str
    model_id: str


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_tokens: int = 2048
    max_workers: int = 8
    rollouts_per_condition: Dict[str, int] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    backend: str = "anthropic"
    model_id: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 512
    max_workers: int = 8
    include_context: bool = False


@dataclass
class ValidationJudgeConfig:
    enabled: bool = False
    backend: str = "openai"
    model_id: str = "gpt-5-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    sample_size: int = 260


@dataclass
class PuzzleConfig:
    countdown_count: int = 60
    fraction_count: int = 40


@dataclass
class WildChatConfig:
    hf_dataset: str = "allenai/WildChat-1M"
    split: str = "train"
    num_prompts: int = 100
    english_only: bool = True
    max_prompt_chars: int = 1200


@dataclass
class OutputConfig:
    dir: str = "results"
    responses_file: str = "responses.jsonl"


@dataclass
class Config:
    seed: int = 1234
    models: List[ModelSpec] = field(default_factory=list)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    validation_judge: ValidationJudgeConfig = field(default_factory=ValidationJudgeConfig)
    puzzles: PuzzleConfig = field(default_factory=PuzzleConfig)
    wildchat: WildChatConfig = field(default_factory=WildChatConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def responses_path(self) -> Path:
        return Path(self.output.dir) / self.output.responses_file


def _coerce(cls, data: Dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys gracefully."""
    if not data:
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | os.PathLike) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        seed=raw.get("seed", 1234),
        models=[ModelSpec(**m) for m in raw.get("models", [])],
        generation=_coerce(GenerationConfig, raw.get("generation", {})),
        judge=_coerce(JudgeConfig, raw.get("judge", {})),
        validation_judge=_coerce(ValidationJudgeConfig, raw.get("validation_judge", {})),
        puzzles=_coerce(PuzzleConfig, raw.get("puzzles", {})),
        wildchat=_coerce(WildChatConfig, raw.get("wildchat", {})),
        output=_coerce(OutputConfig, raw.get("output", {})),
    )
