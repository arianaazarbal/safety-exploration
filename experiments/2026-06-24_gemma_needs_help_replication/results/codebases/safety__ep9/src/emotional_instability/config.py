"""Config loading and lightweight schema access.

The YAML config (``config/config.yaml``) is the single source of truth for
models, sampling, eval budgets, and training hyperparameters. We keep this a
thin wrapper (dict + dotted access) rather than a rigid schema so that the many
knobs the paper exposes stay easy to override from the CLI.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


@dataclass
class ModelSpec:
    """Resolved description of one model in scope."""

    name: str
    backend: str          # "hf" | "api"
    role: str             # "instruct" | "base"
    family: str           # "gemma" | "gemini"
    hf_id: str | None = None
    api_id: str | None = None
    disable_thinking: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Config:
    """Dict-backed config with dotted-key access and CLI overrides."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None,
             profile: str | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls(data)
        for ov in overrides or []:
            key, _, value = ov.partition("=")
            cfg.set(key.strip(), _coerce(value.strip()))
        if profile == "smoke":
            cfg._apply_smoke_profile()
        return cfg

    def _apply_smoke_profile(self) -> None:
        """Scale every sampling budget down for a fast end-to-end smoke test."""
        scale = self.get("eval.smoke_scale", 0.01)
        for cat, spec in self.get("eval.categories", {}).items():
            spec["n_responses"] = max(spec["turns"], int(math.ceil(spec["n_responses"] * scale)))
        self.set("prefill.n_source_responses", 2)
        self.set("prefill.n_numeric_sources", 1)
        self.set("prefill.n_text_sources", 1)
        self.set("prefill.continuations_per_prefill", 4)
        self.set("calm_data.n_target_pairs", 8)
        self.set("calm_data.n_sft_samples", 16)
        self.set("calm_data.n_sft_mix_instruct", 8)
        self.set("petri.transcripts_per_emotion", 1)
        self.set("capabilities.max_samples_per_benchmark", 8)

    # -- access ------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- model resolution --------------------------------------------------
    def model_spec(self, name: str) -> ModelSpec:
        models = self.get("models", {})
        if name not in models:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(models)}")
        m = models[name]
        known = {"backend", "hf_id", "api_id", "role", "family", "disable_thinking"}
        return ModelSpec(
            name=name,
            backend=m["backend"],
            role=m.get("role", "instruct"),
            family=m.get("family", "unknown"),
            hf_id=m.get("hf_id"),
            api_id=m.get("api_id"),
            disable_thinking=m.get("disable_thinking", False),
            extra={k: v for k, v in m.items() if k not in known},
        )

    def section_models(self, section: str) -> list[str]:
        return self.get(f"sections.{section}", [])

    # -- paths -------------------------------------------------------------
    def path(self, key: str) -> Path:
        p = REPO_ROOT / self.get(f"paths.{key}", key)
        p.mkdir(parents=True, exist_ok=True)
        return p


def _coerce(value: str) -> Any:
    """Best-effort coercion of CLI override strings to native types."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") or value.startswith("{"):
        try:
            return yaml.safe_load(value)
        except Exception:
            return value
    return value


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"Environment variable {var} is required but not set. "
            "Export it before running (see README.md)."
        )
    return val
