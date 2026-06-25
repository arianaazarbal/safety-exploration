"""Configuration loading helpers.

Loads ``config/models.yaml`` and ``config/eval.yaml`` and exposes typed-ish
accessors. Kept deliberately small: configs are plain dicts so they round-trip
cleanly to the run manifests written alongside results.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of this package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"           # generated datasets (calm data, DPO pairs, ...)
RESULTS_DIR = REPO_ROOT / "results"     # rollouts, scores, figures
ARTIFACTS_DIR = REPO_ROOT / "artifacts"  # trained LoRA adapters


@lru_cache(maxsize=None)
def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=None)
def models_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "models.yaml")


@lru_cache(maxsize=None)
def eval_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "eval.yaml")


@dataclass(frozen=True)
class ModelSpec:
    """A single model entry from ``models.yaml`` (participant or infrastructure)."""

    name: str
    backend: str                       # hf_local | openrouter | anthropic
    ref: str                           # hf_id / api_id
    family: str = "infra"
    kind: str = "instruct"             # instruct | base
    supports_prefill: bool = False
    finetunable: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.backend == "hf_local"


def get_participant(name: str) -> ModelSpec:
    cfg = models_config()["participants"][name]
    return ModelSpec(
        name=name,
        backend=cfg["backend"],
        ref=cfg.get("hf_id") or cfg.get("api_id"),
        family=cfg.get("family", "infra"),
        kind=cfg.get("kind", "instruct"),
        supports_prefill=cfg.get("supports_prefill", False),
        finetunable=cfg.get("finetunable", False),
        extra_body=cfg.get("extra_body", {}) or {},
    )


def get_infrastructure(role: str) -> ModelSpec:
    cfg = models_config()["infrastructure"][role]
    return ModelSpec(
        name=role,
        backend=cfg["backend"],
        ref=cfg.get("api_id"),
        extra_body=cfg.get("extra_body", {}) or {},
    )


def list_participants() -> list[str]:
    return list(models_config()["participants"].keys())


def generation_defaults() -> dict[str, Any]:
    return dict(models_config()["generation_defaults"])


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
