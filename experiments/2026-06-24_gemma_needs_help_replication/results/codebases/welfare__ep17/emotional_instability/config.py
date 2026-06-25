"""Typed access to config.yaml plus small helpers shared across scripts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"


@dataclass
class ModelSpec:
    name: str
    backend: str           # "hf" | "openrouter"
    family: str            # "gemma" | "gemini"
    kind: str              # "instruct" | "base"
    roles: list[str]
    hf_id: str | None = None
    openrouter_id: str | None = None
    # Optional PEFT/LoRA adapter applied on top of `hf_id` (the DPO/SFT models).
    adapter_path: str | None = None

    @property
    def ident(self) -> str:
        """The backend-specific identifier used to load / call the model."""
        if self.backend == "hf":
            if not self.hf_id:
                raise ValueError(f"model {self.name} has backend=hf but no hf_id")
            return self.hf_id
        if self.backend == "openrouter":
            if not self.openrouter_id:
                raise ValueError(
                    f"model {self.name} has backend=openrouter but no openrouter_id"
                )
            return self.openrouter_id
        raise ValueError(f"unknown backend {self.backend!r} for model {self.name}")


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    # -- models ---------------------------------------------------------------
    @property
    def models(self) -> dict[str, ModelSpec]:
        out: dict[str, ModelSpec] = {}
        for name, m in self.raw["models"].items():
            out[name] = ModelSpec(
                name=name,
                backend=m["backend"],
                family=m["family"],
                kind=m["kind"],
                roles=list(m.get("roles", [])),
                hf_id=m.get("hf_id"),
                openrouter_id=m.get("openrouter_id"),
                adapter_path=m.get("adapter_path"),
            )
        return out

    def model(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError:
            raise KeyError(
                f"model {name!r} not in config; known: {sorted(self.models)}"
            ) from None

    def models_with_role(self, role: str) -> list[ModelSpec]:
        return [m for m in self.models.values() if role in m.roles]

    # -- generic section access ----------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- evaluation -----------------------------------------------------------
    def scaled_samples(self, condition_name: str) -> int:
        """Per-condition sample count after applying the global `scale` factor.

        `scale < 1` gives a cheap smoke run; we round up so every condition
        keeps at least one sample.
        """
        ev = self.raw["evaluation"]
        n = ev["conditions"][condition_name]["samples"]
        scale = float(ev.get("scale", 1.0))
        return max(1, math.ceil(n * scale))

    # -- paths ----------------------------------------------------------------
    def path_for(self, key: str) -> Path:
        p = _REPO_ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else Path(os.environ.get("EI_CONFIG", _DEFAULT_CONFIG))
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=cfg_path)
