"""Shared bootstrap for experiment scripts: put `src/` on the path and parse the
common CLI flags (--config / --set / --profile / --models)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from emotional_instability.config import Config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--set", action="append", default=[], metavar="key=value",
                   help="Override a config value (dotted key). Repeatable.")
    p.add_argument("--profile", default=None, choices=["smoke"],
                   help="Use a reduced sampling profile for a fast end-to-end test.")
    p.add_argument("--models", default=None,
                   help="Comma-separated model names (overrides the section default).")
    return p


def load_config(args: argparse.Namespace) -> Config:
    return Config.load(path=args.config, overrides=args.set, profile=args.profile)


def resolve_models(args: argparse.Namespace, cfg: Config, section: str) -> list[str]:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    return cfg.section_models(section)
