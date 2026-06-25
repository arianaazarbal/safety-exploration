"""Shared CLI plumbing for the scripts.

Adds ``src`` to the path so scripts run without installation, and provides a
standard argument parser for ``--config`` and ``--full`` (disable dev_mode).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from emotional_instability.config import load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to YAML config (default: config/default.yaml)")
    p.add_argument("--full", action="store_true",
                   help="disable dev_mode and use the full paper sample sizes (4000/model). "
                        "This runs the FULL distress-inducing protocol -- see DESIGN.md welfare notes.")
    return p


def cfg_from_args(args) -> dict:
    overrides = {"run": {"dev_mode": not args.full}} if args.full else None
    return load_config(args.config, overrides=overrides)
