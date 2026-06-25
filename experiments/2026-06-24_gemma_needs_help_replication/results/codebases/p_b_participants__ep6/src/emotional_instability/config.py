"""Configuration loading.

A thin wrapper over the YAML in ``config/default.yaml``. We deliberately keep the
config as plain nested dicts (accessed via ``cfg["..."]``) rather than a rigid
schema: the replication has many knobs and a dict keeps scripts terse while the
YAML file remains the single, commented source of truth.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the YAML config, optionally deep-merging ``overrides`` on top."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def sample_sizes(cfg: dict) -> dict[str, int]:
    """Return the active per-category sample sizes, honouring ``dev_mode``."""
    key = "dev_sample_sizes" if cfg["run"].get("dev_mode", True) else "sample_sizes"
    return cfg[key]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
