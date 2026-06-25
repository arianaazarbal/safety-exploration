"""Configuration loading. Reads `config.yaml` from the repo root (override with
the EI_CONFIG environment variable)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("EI_CONFIG", REPO_ROOT / "config.yaml"))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def eval_counts(cfg: dict) -> dict[str, int]:
    """Conversation counts for the active preset (Section 2)."""
    preset = cfg.get("preset", "smoke")
    return cfg["eval"]["presets"][preset]


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve one of the configured output directories, creating it."""
    p = REPO_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
