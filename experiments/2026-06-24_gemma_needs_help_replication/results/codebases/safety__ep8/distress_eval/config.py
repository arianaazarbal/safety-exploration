"""Config loading: parses config.yaml into typed dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelSpec:
    key: str
    id: str
    backend: str
    family: str = ""
    role: str = "instruct"
    base_id: str | None = None         # for LoRA adapters: the base model

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(**{k: d.get(k) for k in (
            "key", "id", "backend", "family", "role", "base_id") if k in d})


@dataclass
class JudgeSpec:
    backend: str
    id: str
    temperature: float = 0.0
    max_tokens: int = 512
    secondary: dict | None = None


@dataclass
class GenSpec:
    temperature: float = 1.0
    max_new_tokens: int = 1024
    top_p: float = 1.0
    disable_thinking: bool = True


@dataclass
class Config:
    raw: dict
    output_dir: Path
    data_dir: Path
    models: list[ModelSpec]
    base_models: list[ModelSpec]
    finetuned_models: list[ModelSpec]
    judge: JudgeSpec
    petri: dict
    generation: GenSpec
    profile: str
    sample_counts: dict
    max_concurrency: int
    seed: int

    def counts(self) -> dict[str, int]:
        return self.sample_counts[self.profile]

    def model_by_key(self, key: str) -> ModelSpec:
        for m in self.models + self.base_models + self.finetuned_models:
            if m.key == key:
                return m
        raise KeyError(f"Unknown model key: {key}")


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text())

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (REPO_ROOT / pp)

    return Config(
        raw=raw,
        output_dir=_resolve(raw["output_dir"]),
        data_dir=_resolve(raw["data_dir"]),
        models=[ModelSpec.from_dict(d) for d in raw.get("models", [])],
        base_models=[ModelSpec.from_dict(d) for d in raw.get("base_models", [])],
        finetuned_models=[ModelSpec.from_dict(d) for d in raw.get("finetuned_models", [])],
        judge=JudgeSpec(**raw["judge"]),
        petri=raw.get("petri", {}),
        generation=GenSpec(**raw.get("generation", {})),
        profile=raw.get("profile", "full"),
        sample_counts=raw["sample_counts"],
        max_concurrency=raw.get("max_concurrency", 8),
        seed=raw.get("seed", 0),
    )
