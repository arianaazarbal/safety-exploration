"""Configuration loading.

Thin wrapper over the YAML config that resolves the active scale, expands
per-condition sample counts, and provides typed-ish accessors. We deliberately
keep this as plain dicts (with helpers) rather than a rigid schema so that the
config stays easy to override from the command line / experiments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    # --- generic access -----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # --- models / judges ----------------------------------------------------
    def model(self, name: str) -> dict[str, Any]:
        if name in self.raw["models"]:
            return {"name": name, **self.raw["models"][name]}
        raise KeyError(f"Unknown model '{name}'. Known: {list(self.raw['models'])}")

    def judge(self, role: str) -> dict[str, Any]:
        return {"name": role, **self.raw["judges"][role]}

    def target_models(self) -> list[str]:
        return [n for n, m in self.raw["models"].items() if m.get("is_target")]

    # --- sampling counts ----------------------------------------------------
    def category_response_count(self, category: str) -> int:
        """Paper response count for a category, scaled. Always >= 1 if nonzero."""
        base = self.raw["sampling"]["paper_counts"].get(category, 0)
        scale = float(self.raw["sampling"]["scale"])
        if base == 0:
            return 0
        return max(1, round(base * scale))

    # --- paths --------------------------------------------------------------
    def path_for(self, key: str) -> Path:
        p = REPO_ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=path)
