"""Configuration loading.

Config is YAML (see config.yaml / config.full.yaml). We keep it as light
dataclasses so the runner has typed access and defaults live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .conditions import ALL_CONDITIONS


@dataclass
class ModelSpec:
    id: str
    provider: str = "google"
    extra: dict = field(default_factory=dict)

    def to_provider_spec(self) -> dict:
        return {"id": self.id, "provider": self.provider, **self.extra}


@dataclass
class JudgeConfig:
    id: str = "claude-sonnet-4-5"
    provider: str = "anthropic"
    temperature: float = 0.0
    max_tokens: int = 256
    use_context: bool = False
    extra: dict = field(default_factory=dict)

    def to_provider_spec(self) -> dict:
        return {"id": self.id, "provider": self.provider, **self.extra}


@dataclass
class Config:
    models: list[ModelSpec]
    judge: JudgeConfig
    rollouts_per_condition: dict[str, int]
    temperature: float = 1.0
    max_tokens: int = 2048
    concurrency: int = 8
    seed: int = 0
    output_dir: str = "results"
    wildchat_dataset: str = "allenai/WildChat-1M"
    verify_puzzles: bool = True

    def rollouts_for(self, condition: str) -> int:
        return int(self.rollouts_per_condition.get(condition, 0))


def _parse_models(raw: list[dict]) -> list[ModelSpec]:
    models = []
    for m in raw:
        known = {"id", "provider"}
        extra = {k: v for k, v in m.items() if k not in known}
        models.append(
            ModelSpec(id=m["id"], provider=m.get("provider", "google"), extra=extra)
        )
    return models


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    models = _parse_models(data["models"])

    jraw = data.get("judge", {})
    known_judge = {"id", "provider", "temperature", "max_tokens", "use_context"}
    judge = JudgeConfig(
        id=jraw.get("id", "claude-sonnet-4-5"),
        provider=jraw.get("provider", "anthropic"),
        temperature=jraw.get("temperature", 0.0),
        max_tokens=jraw.get("max_tokens", 256),
        use_context=jraw.get("use_context", False),
        extra={k: v for k, v in jraw.items() if k not in known_judge},
    )

    scale = data.get("scale", {})
    rpc = dict(scale.get("rollouts_per_condition", {}))
    # Default any unspecified condition to 0 (skipped) but warn-friendly: keep keys.
    for cond in ALL_CONDITIONS:
        rpc.setdefault(cond, 0)

    sampling = data.get("sampling", {})

    return Config(
        models=models,
        judge=judge,
        rollouts_per_condition=rpc,
        temperature=sampling.get("temperature", 1.0),
        max_tokens=sampling.get("max_tokens", 2048),
        concurrency=data.get("concurrency", 8),
        seed=data.get("seed", 0),
        output_dir=data.get("output_dir", "results"),
        wildchat_dataset=data.get("wildchat", {}).get("dataset", "allenai/WildChat-1M"),
        verify_puzzles=data.get("verify_puzzles", True),
    )
