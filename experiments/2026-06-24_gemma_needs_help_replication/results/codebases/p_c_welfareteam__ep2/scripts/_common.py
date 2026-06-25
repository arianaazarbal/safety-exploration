"""Shared CLI bootstrap for the scripts.

Adds ``src/`` to the path (so scripts run without installing the package) and
provides a uniform config loader with ``--models`` / ``--overrides`` flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--overrides", default=None)


def load(args) -> "object":
    from gemma_distress.config import load_config
    from gemma_distress.utils.seeding import seed_everything

    cfg = load_config(args.models, args.overrides)
    seed_everything(cfg.eval.seed)
    return cfg
