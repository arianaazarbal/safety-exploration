#!/usr/bin/env python3
"""CLI for the distress-elicitation replication.

Examples
--------
    # Full run with defaults (Gemma + Gemini via OpenRouter), then aggregate:
    python run_eval.py all

    # Use a config file:
    python run_eval.py all --config config.yaml

    # Quick smoke test (~1% of the responses):
    python run_eval.py all --smoke

    # Just (re)generate, just aggregate, or just the reliability cross-judge:
    python run_eval.py run --config config.yaml
    python run_eval.py aggregate --config config.yaml
    python run_eval.py cross-judge --config config.yaml

Environment
-----------
    ANTHROPIC_API_KEY    required for the judge
    OPENROUTER_API_KEY   required for OpenRouter target models / cross-judge
"""

from __future__ import annotations

import argparse
import sys

from distress_eval.config import default_config, load_config, RunConfig
from distress_eval import runner, aggregate as agg


def _load(args) -> RunConfig:
    cfg = load_config(args.config) if args.config else default_config()
    if getattr(args, "smoke", False):
        cfg.scale = 0.01
        cfg.run_name = f"{cfg.run_name}-smoke"
    if getattr(args, "scale", None) is not None:
        cfg.scale = args.scale
    if getattr(args, "run_name", None):
        cfg.run_name = args.run_name
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("run", "aggregate", "cross-judge", "all"):
        p = sub.add_parser(name)
        p.add_argument("--config", help="path to a YAML config (optional)")
        p.add_argument("--run-name", dest="run_name", help="override run name")
        p.add_argument("--scale", type=float, help="override rollout scale (1.0 = paper)")
        p.add_argument("--smoke", action="store_true", help="tiny run for testing")
        if name in ("aggregate", "all"):
            p.add_argument("--no-figures", action="store_true",
                           help="skip matplotlib figure generation")

    args = parser.parse_args(argv)
    cfg = _load(args)

    if args.cmd == "run":
        runner.run(cfg)
    elif args.cmd == "aggregate":
        agg.aggregate(cfg, make_figures=not args.no_figures)
    elif args.cmd == "cross-judge":
        runner.run_cross_judge(cfg)
    elif args.cmd == "all":
        runner.run(cfg)
        if cfg.judge.cross_provider:
            runner.run_cross_judge(cfg)
        agg.aggregate(cfg, make_figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
