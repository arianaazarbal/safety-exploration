"""Load and validate run configuration from YAML, plus environment (.env)."""

from __future__ import annotations

import os
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load the YAML config and environment variables (.env if present)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass  # dotenv is a convenience; env vars may already be set

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    for key in ("run", "models", "generation", "judge", "sampling"):
        if key not in cfg:
            raise ValueError(f"config missing required section: {key!r}")
    if not cfg["models"]:
        raise ValueError("config.models is empty")
    for m in cfg["models"]:
        for k in ("name", "backend", "model_id"):
            if k not in m:
                raise ValueError(f"model entry missing {k!r}: {m}")
        if m["backend"] not in ("openrouter", "vllm"):
            raise ValueError(f"unknown backend {m['backend']!r} for model {m['name']!r}")
    preset = cfg["sampling"].get("preset")
    if preset not in ("paper", "smoke", "custom"):
        raise ValueError(f"sampling.preset must be paper|smoke|custom, got {preset!r}")


def run_paths(cfg: dict) -> dict[str, str]:
    out = cfg["run"]["out_dir"]
    return {
        "out_dir": out,
        "rollouts": os.path.join(out, "rollouts.jsonl"),
        "scores": os.path.join(out, "scores.jsonl"),
        "metrics": os.path.join(out, "metrics"),
        "config_snapshot": os.path.join(out, "config_snapshot.yaml"),
    }
