"""Shared CLI helpers for the orchestration scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` importable when running scripts directly.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from distress_eval.config import load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="Path to YAML config (default: configs/default.yaml)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of participant model keys (default: all participants)")
    p.add_argument("--full", action="store_true",
                   help="Use the paper's full sampling plan (eval.full_counts) instead of the small default")
    return p


def load(args):
    cfg = load_config(args.config)
    if getattr(args, "full", False):
        cfg.eval.counts = dict(cfg.eval.full_counts)
    return cfg


def resolve_models(cfg, requested, *, default_role="participant"):
    if requested:
        return list(requested)
    return [k for k, m in cfg.models.items() if m.role == default_role]
