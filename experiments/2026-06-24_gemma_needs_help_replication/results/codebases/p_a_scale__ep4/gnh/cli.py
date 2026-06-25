"""Shared CLI plumbing for the scripts: config loading, logging, scaling.

`--scale` multiplies all sampling budgets (conversation counts, continuations,
transcripts) so you can run a cheap pilot (e.g. --scale 0.02) before committing
to the full multi-week sweep, without editing the YAML.
"""
from __future__ import annotations

import argparse
import math

from gnh.config import Config, load_config
from gnh.logging_utils import setup_logging
from gnh.models.registry import BackendRegistry


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply all sampling budgets (use <1 for a pilot).")
    p.add_argument("--models", nargs="*", default=None,
                   help="Override the target model list for this run.")
    return p


def _scaled(n: int, scale: float) -> int:
    return max(1, int(math.ceil(n * scale)))


def apply_scale(cfg: Config, scale: float) -> None:
    if scale == 1.0:
        return
    for cat in cfg.eval.get("categories", {}).values():
        cat["n_conversations"] = _scaled(cat.get("n_conversations", 1), scale)
    if "validation" in cfg.eval:
        cfg.eval["validation"]["n_samples"] = _scaled(cfg.eval["validation"].get("n_samples", 1), scale)
    cfg.prefill["continuations_per_prefill"] = _scaled(cfg.prefill.get("continuations_per_prefill", 1), scale)
    cfg.training["calm_data"]["n_conversations"] = _scaled(cfg.training["calm_data"].get("n_conversations", 1), scale)
    cfg.petri["transcripts_per_emotion"] = _scaled(cfg.petri.get("transcripts_per_emotion", 1), scale)


def setup(args) -> tuple[Config, BackendRegistry]:
    cfg = load_config(args.config)
    apply_scale(cfg, getattr(args, "scale", 1.0))
    if getattr(args, "models", None):
        cfg.target_models = args.models
    setup_logging(cfg.output_path, cfg.run.log_level)
    return cfg, BackendRegistry(cfg)
