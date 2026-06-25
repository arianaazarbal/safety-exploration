"""Lightweight config loading and project paths."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels above this file's package dir (src/emotional_instability/..)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        # allow relative-to-root or relative-to-config
        candidates = [PROJECT_ROOT / path, CONFIG_DIR / path]
        path = next((c for c in candidates if c.exists()), candidates[0])
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models_config(path: str | Path = "models.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_eval_config(path: str | Path = "eval.yaml") -> "EvalConfig":
    return EvalConfig(load_yaml(path))


def load_training_config(path: str | Path = "training.yaml") -> dict[str, Any]:
    return load_yaml(path)


@dataclass
class EvalConfig:
    raw: dict[str, Any]

    def scaled_n(self, condition_key: str) -> int:
        """Per-condition sample count after applying sampling.scale."""
        scale = float(self.raw["sampling"]["scale"])
        base = int(self.raw["conditions"][condition_key]["n_responses"])
        if scale <= 0:
            return 0
        return max(1, math.ceil(base * scale))

    @property
    def conditions(self) -> dict[str, Any]:
        return self.raw["conditions"]

    @property
    def sampling(self) -> dict[str, Any]:
        return self.raw["sampling"]

    @property
    def output_dir(self) -> Path:
        return PROJECT_ROOT / self.raw.get("output_dir", "outputs")

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"Environment variable {var} is required for this API client but is unset."
        )
    return val
