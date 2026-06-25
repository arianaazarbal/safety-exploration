"""Configuration loading and lightweight schema objects.

All experiment parameters live in the YAML files under ``config/``; this module
just loads them and exposes a couple of typed accessors so the rest of the code
doesn't sprinkle dictionary lookups everywhere.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of the package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=None)
def models_config() -> dict[str, Any]:
    return _load_yaml("models.yaml")


@lru_cache(maxsize=None)
def eval_config() -> dict[str, Any]:
    return _load_yaml("eval.yaml")


@lru_cache(maxsize=None)
def training_config() -> dict[str, Any]:
    return _load_yaml("training.yaml")


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a model (subject or infrastructure)."""

    name: str
    backend: str                  # "local" | "openrouter"
    family: str
    kind: str = "instruct"        # "instruct" | "base"
    hf_id: str | None = None
    api_id: str | None = None
    supports_prefill: bool = False
    supports_hidden_states: bool = False
    thinking: bool = False
    temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.backend == "local"


def _spec_from_dict(name: str, d: dict[str, Any]) -> ModelSpec:
    known = {
        "backend", "family", "kind", "hf_id", "api_id",
        "supports_prefill", "supports_hidden_states", "thinking", "temperature",
    }
    extra = {k: v for k, v in d.items() if k not in known}
    return ModelSpec(
        name=name,
        backend=d["backend"],
        family=d.get("family", "unknown"),
        kind=d.get("kind", "instruct"),
        hf_id=d.get("hf_id"),
        api_id=d.get("api_id"),
        supports_prefill=d.get("supports_prefill", False),
        supports_hidden_states=d.get("supports_hidden_states", False),
        thinking=d.get("thinking", False),
        temperature=d.get("temperature"),
        extra=extra,
    )


def get_model_spec(name: str) -> ModelSpec:
    """Resolve a subject model by its registry name."""
    cfg = models_config()
    if name not in cfg["models"]:
        raise KeyError(
            f"Unknown model '{name}'. Known: {sorted(cfg['models'])}"
        )
    return _spec_from_dict(name, cfg["models"][name])


def get_infra_spec(role: str, slot: str = "primary") -> ModelSpec:
    """Resolve a judge/auditor/labeller model.

    ``role`` is one of: 'judge', 'prefill_labelling', 'petri'.
    ``slot`` is the sub-key (e.g. 'primary', 'secondary', 'auditor', 'judge',
    'onset_labeller', 'paraphraser').
    """
    cfg = models_config()
    if role not in cfg:
        raise KeyError(f"Unknown infra role '{role}'")
    if slot not in cfg[role]:
        raise KeyError(f"Unknown slot '{slot}' for role '{role}'")
    return _spec_from_dict(f"{role}.{slot}", cfg[role][slot])


def subject_models() -> list[str]:
    """All in-scope subject model names."""
    return list(models_config()["models"].keys())


def n_conversations_for(category: str) -> int:
    """Derive conversation count from target_responses / turns (Appendix B).

    We score every assistant turn, so a T-turn conversation yields T responses.
    """
    cat = eval_config()["categories"][category]
    return math.ceil(cat["target_responses"] / cat["turns"])


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
