"""Shared argument parsing / path bootstrap for the CLI scripts."""
from __future__ import annotations

import argparse
import os
import sys

# Make the package importable when scripts are run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma_distress.config import load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="Path to a YAML config file.")
    p.add_argument("--output-dir", default=None,
                   help="Override config.output_dir.")
    return p


def make_config(args):
    cfg = load_config(args.config)
    if getattr(args, "output_dir", None):
        cfg.output_dir = args.output_dir
    return cfg
