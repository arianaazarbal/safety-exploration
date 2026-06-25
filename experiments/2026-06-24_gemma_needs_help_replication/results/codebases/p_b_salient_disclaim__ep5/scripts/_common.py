"""Shared helpers for the CLI scripts (path defaults, model building)."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when scripts are run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemma_distress.config import CONFIG_DIR, ExperimentConfig, ModelRegistry  # noqa: E402
from gemma_distress.models import build_model  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def load(exp_path: str | None = None):
    registry = ModelRegistry()
    exp = ExperimentConfig.load(exp_path or (CONFIG_DIR / "experiment.yaml"))
    return registry, exp


def model(registry: ModelRegistry, name: str):
    return build_model(registry.get(name))


def outdir(*parts: str) -> Path:
    p = OUTPUTS.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
