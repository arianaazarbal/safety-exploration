"""Load and validate config.yaml, and construct clients from it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .clients import ChatClient, build_client


@dataclass
class TargetModel:
    name: str            # internal short name, e.g. "gemma-3-27b-it"
    provider: str        # key into providers map
    model: str           # provider-specific model id, e.g. "google/gemma-3-27b-it"
    disable_thinking: bool = False


@dataclass
class JudgeConfig:
    provider: str
    model: str
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass
class Config:
    raw: dict[str, Any]
    providers: dict[str, dict[str, Any]]
    targets: list[TargetModel]
    judge: JudgeConfig
    temperature: float
    max_tokens: int
    concurrency: int
    max_retries: int
    base_delay: float
    output_dir: Path
    seed: int
    scale: float
    score_all_turns: bool
    allow_wildchat_download: bool

    # ---- client factories -------------------------------------------------

    def _extra_body_for(self, target: TargetModel) -> dict[str, Any] | None:
        """Provider-specific request extras. We disable hidden reasoning where the
        provider supports it (paper: 'we set thinking to be false via the API')."""
        if not target.disable_thinking:
            return None
        provider_type = self.providers[target.provider]["type"]
        if provider_type == "openai_compatible":
            # OpenRouter understands `reasoning`; Google's OpenAI-compat understands
            # `reasoning_effort`/thinking config. We send both-compatible hints; unknown
            # keys are ignored by providers that don't support them.
            return {"reasoning": {"enabled": False}}
        return None

    def target_client(self, target: TargetModel) -> ChatClient:
        return build_client(
            self.providers[target.provider],
            target.model,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            default_extra_body=self._extra_body_for(target),
        )

    def judge_client(self) -> ChatClient:
        return build_client(
            self.providers[self.judge.provider],
            self.judge.model,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
        )

    def target_by_name(self, name: str) -> TargetModel:
        for t in self.targets:
            if t.name == name:
                return t
        raise KeyError(f"no target model named {name!r}; have {[t.name for t in self.targets]}")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())

    providers = raw["providers"]
    targets = [
        TargetModel(
            name=t["name"],
            provider=t["provider"],
            model=t["model"],
            disable_thinking=bool(t.get("disable_thinking", False)),
        )
        for t in raw["targets"]
    ]
    j = raw["judge"]
    judge = JudgeConfig(
        provider=j["provider"],
        model=j["model"],
        max_tokens=int(j.get("max_tokens", 512)),
        temperature=float(j.get("temperature", 0.0)),
    )
    gen = raw.get("generation", {})
    run = raw.get("run", {})
    samp = raw.get("sampling", {})

    cfg = Config(
        raw=raw,
        providers=providers,
        targets=targets,
        judge=judge,
        temperature=float(gen.get("temperature", 1.0)),
        max_tokens=int(gen.get("max_tokens", 2048)),
        concurrency=int(run.get("concurrency", 8)),
        max_retries=int(run.get("max_retries", 5)),
        base_delay=float(run.get("base_delay", 2.0)),
        output_dir=Path(run.get("output_dir", "results")),
        seed=int(run.get("seed", 0)),
        scale=float(samp.get("scale", 1.0)),
        score_all_turns=bool(run.get("score_all_turns", True)),
        allow_wildchat_download=bool(run.get("allow_wildchat_download", True)),
    )

    # Validate provider references resolve.
    for t in targets:
        if t.provider not in providers:
            raise ValueError(f"target {t.name!r} references unknown provider {t.provider!r}")
    if judge.provider not in providers:
        raise ValueError(f"judge references unknown provider {judge.provider!r}")
    return cfg
