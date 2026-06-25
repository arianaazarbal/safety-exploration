"""Shared argument-parsing / config-loading helpers for the CLI scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_eval_config, output_root
from ..utils.seeding import seed_everything


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval-config", default="evaluation.yaml",
                        help="Evaluation config path (relative to configs/ or absolute).")
    parser.add_argument("--sample-fraction", type=float, default=None,
                        help="Scale sample counts (e.g. 0.02 for a smoke test).")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--out", default=None, help="Output directory (default: outputs/).")


def load_eval_cfg(args: argparse.Namespace) -> dict:
    cfg = load_eval_config(args.eval_config)
    if args.sample_fraction is not None:
        cfg["sample_fraction"] = args.sample_fraction
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed_everything(cfg.get("seed", 0))
    return cfg


def out_dir(args: argparse.Namespace, subdir: str) -> Path:
    base = output_root() if args.out is None else Path(args.out)
    d = base / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d
