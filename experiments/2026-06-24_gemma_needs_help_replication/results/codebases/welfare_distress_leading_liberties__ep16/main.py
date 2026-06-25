"""CLI entrypoint for the distress-elicitation replication.

Examples:
  python main.py generate --scale pilot
  python main.py score
  python main.py analyze
  python main.py all --scale medium --models gemma-3-27b-it gemini-2.5-flash

Stages share data via data/ on disk, so you can re-run `score`/`analyze`
without regenerating, or re-judge with a different --judge-model.
"""
from __future__ import annotations

import argparse

from config import Config, GEN_MODELS, SCALE_PRESETS, DEFAULT_MODELS
from generate import generate
from judge import score
from analyze import analyze


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if getattr(args, "scale", None):
        cfg = cfg.with_scale(args.scale)
    if getattr(args, "models", None):
        cfg.models = args.models
    if getattr(args, "judge_model", None):
        cfg.judge_model = args.judge_model
    if getattr(args, "judge_provider", None):
        cfg.judge_provider = args.judge_provider
    if getattr(args, "concurrency", None):
        cfg.max_concurrency = args.concurrency
    if getattr(args, "max_tokens", None):
        cfg.max_tokens = args.max_tokens
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    return cfg


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scale", choices=list(SCALE_PRESETS), help="sampling scale preset")
    p.add_argument("--models", nargs="+", choices=list(GEN_MODELS),
                   help=f"subset of models (default: {DEFAULT_MODELS})")
    p.add_argument("--judge-model", dest="judge_model", help="override judge model id")
    p.add_argument("--judge-provider", dest="judge_provider",
                   choices=["anthropic", "openrouter"], help="judge backend")
    p.add_argument("--concurrency", type=int, help="max in-flight API calls")
    p.add_argument("--max-tokens", dest="max_tokens", type=int,
                   help="max tokens per assistant turn")
    p.add_argument("--seed", type=int, help="sampling seed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "score", "analyze", "all"):
        sp = sub.add_parser(name, help=f"run the {name} stage")
        add_common(sp)

    args = parser.parse_args()
    cfg = build_config(args)

    if args.cmd in ("generate", "all"):
        generate(cfg)
    if args.cmd in ("score", "all"):
        score(cfg)
    if args.cmd in ("analyze", "all"):
        analyze(cfg)


if __name__ == "__main__":
    main()
