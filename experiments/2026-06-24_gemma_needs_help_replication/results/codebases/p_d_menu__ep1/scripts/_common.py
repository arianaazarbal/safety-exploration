"""Shared helpers for the run scripts: make the src/ package importable and load
config with optional welfare overrides."""
from __future__ import annotations

import argparse
import os
import sys

# Make src/ importable when running scripts directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from emotional_instability.config import load_config  # noqa: E402


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-welfare", action="store_true",
                        help="Disable the welfare-protection layer (control run).")
    parser.add_argument("--out", default=None, help="Override output directory.")


def get_config(args) -> "object":
    cfg = load_config()
    if getattr(args, "no_welfare", False):
        cfg.welfare["enabled"] = False
    return cfg
