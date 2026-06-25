"""Shared helpers for the runnable scripts."""

from __future__ import annotations

import argparse
import os
import sys

# Allow `python scripts/foo.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distress.config import Config, load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="optional YAML config override")
    p.add_argument("--output-dir", default="runs", help="root output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny counts for a quick end-to-end smoke run",
    )
    return p


def make_config(args) -> Config:
    cfg = load_config(args.config)
    cfg.output_dir = args.output_dir
    cfg.seed = args.seed
    if getattr(args, "smoke", False):
        cfg.counts.impossible_numeric = 8
        cfg.counts.triggers = 4
        cfg.counts.tones = 6
        cfg.counts.extended = 2
        cfg.counts.wildchat = 4
        cfg.judge.cross_judge_n = 8
        cfg.prefill.continuations_per_prefill = 4
        cfg.petri.transcripts_per_emotion = 1
        cfg.petri.max_turns = 4
        cfg.capabilities.max_examples_per_benchmark = 8
    return cfg


def run_dir(cfg: Config, name: str) -> str:
    path = os.path.join(cfg.output_dir, name)
    os.makedirs(path, exist_ok=True)
    return path
