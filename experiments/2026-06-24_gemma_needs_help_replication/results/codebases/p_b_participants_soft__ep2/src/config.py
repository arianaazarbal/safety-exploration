"""Central configuration and model registry.

Loads ``config.yaml`` and exposes typed accessors. Environment variables (API
keys) are loaded from ``.env`` if present. The registry distinguishes
*participants* (the Gemma/Gemini subjects under study) from *infrastructure*
models (Claude judge/auditor, GPT-5-mini validator) which are kept as the paper
specifies.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv optional at import time
    pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@dataclass(frozen=True)
class ModelSpec:
    """A participant model and how to reach it."""

    name: str
    backend: str            # "hf" | "openrouter" | "google"
    family: str             # "gemma" | "gemini"
    role: str               # "instruct" | "base"
    hf_id: str | None = None
    api_id: str | None = None
    # For finetuned variants: a base participant + a LoRA adapter directory.
    base: str | None = None
    adapter_path: str | None = None

    @property
    def is_local(self) -> bool:
        return self.backend == "hf"

    @property
    def supports_prefill(self) -> bool:
        # Only local HF models can be prefilled / probed / trained.
        return self.backend == "hf"


@dataclass(frozen=True)
class InfraSpec:
    provider: str           # "anthropic" | "openai" | "openrouter" | "google"
    model: str


@dataclass
class Config:
    raw: dict[str, Any]
    seed: int
    temperature: float
    max_new_tokens: int
    participants: dict[str, ModelSpec]
    finetuned: dict[str, ModelSpec]
    judge: InfraSpec
    judge_validation: InfraSpec
    onset_labeller: InfraSpec
    paraphraser: InfraSpec
    petri_auditor: InfraSpec
    petri_judge: InfraSpec
    sample_plan: dict[str, int]
    paths: dict[str, str]

    # --- convenience ---------------------------------------------------
    def model(self, name: str) -> ModelSpec:
        if name in self.participants:
            return self.participants[name]
        if name in self.finetuned:
            return self.finetuned[name]
        raise KeyError(f"Unknown model '{name}'. Known: "
                       f"{sorted(self.participants) + sorted(self.finetuned)}")

    def gemma_participants(self) -> list[str]:
        return [n for n, s in self.participants.items() if s.family == "gemma"]

    def gemini_participants(self) -> list[str]:
        return [n for n, s in self.participants.items() if s.family == "gemini"]

    def out(self, *parts: str) -> Path:
        p = ROOT / self.paths.get("outputs_dir", "outputs")
        p = p.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def data(self, *parts: str) -> Path:
        p = ROOT / self.paths.get("data_dir", "data")
        p = p.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _parse_participant(name: str, d: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        name=name,
        backend=d["backend"],
        family=d["family"],
        role=d.get("role", "instruct"),
        hf_id=d.get("hf_id"),
        api_id=d.get("api_id"),
    )


def load_config(path: str | Path = CONFIG_PATH) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    participants = {n: _parse_participant(n, d) for n, d in raw["participants"].items()}

    finetuned: dict[str, ModelSpec] = {}
    for n, d in raw.get("finetuned", {}).items():
        base = participants[d["base"]]
        finetuned[n] = ModelSpec(
            name=n,
            backend="hf",          # adapters are always loaded on a local base
            family=base.family,
            role="instruct",
            hf_id=base.hf_id,
            base=d["base"],
            adapter_path=d["adapter_path"],
        )

    def infra(key: str) -> InfraSpec:
        d = raw[key]
        return InfraSpec(provider=d["provider"], model=d["model"])

    return Config(
        raw=raw,
        seed=raw.get("seed", 0),
        temperature=raw.get("temperature", 1.0),
        max_new_tokens=raw.get("max_new_tokens", 2048),
        participants=participants,
        finetuned=finetuned,
        judge=infra("judge"),
        judge_validation=infra("judge_validation"),
        onset_labeller=infra("onset_labeller"),
        paraphraser=infra("paraphraser"),
        petri_auditor=infra("petri_auditor"),
        petri_judge=infra("petri_judge"),
        sample_plan=raw["sample_plan"],
        paths=raw.get("paths", {}),
    )


# A module-level singleton is convenient; callers may also build their own.
CFG = load_config()
