"""Command-line entry point: ``gnh <command> [opts]``.

Commands map to the paper's experiments. All heavy lifting lives in the library;
this is a thin argparse shell. Nothing here is executed during implementation —
it documents how the pieces compose into a full replication run.
"""

from __future__ import annotations

import argparse

from .config import load_config
from .models.registry import GEMINI_MODELS, GEMMA_MODELS


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Path to YAML config override.")
    p.add_argument("--seed", type=int, default=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gnh", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # distress-eval (Section 2)
    p = sub.add_parser("distress-eval", help="Section 2 distress elicitation.")
    p.add_argument("--model", required=True, help="Target model name (Gemma/Gemini).")
    p.add_argument("--adapter", default=None, help="Optional LoRA adapter path.")
    _add_common(p)

    # validate-judge (Section 2.1)
    p = sub.add_parser("validate-judge", help="GPT-5-mini judge-agreement check.")
    p.add_argument("--model", required=True)
    _add_common(p)

    # list-models
    sub.add_parser("list-models", help="List in-scope target models.")

    args = parser.parse_args(argv)

    if args.command == "list-models":
        print("Gemma (local HF):")
        for k in GEMMA_MODELS:
            print(f"  - {k}")
        print("Gemini (OpenRouter):")
        for k in GEMINI_MODELS:
            print(f"  - {k}")
        return

    cfg = load_config(getattr(args, "config", None))

    if args.command == "distress-eval":
        from .pipeline import run_distress_eval

        summary = run_distress_eval(
            cfg, args.model, adapter_path=args.adapter, seed=args.seed)
        print(f"overall %high = {summary['overall']['pct_high']:.2f}")

    elif args.command == "validate-judge":
        from .pipeline import run_judge_validation

        report = run_judge_validation(cfg, args.model)
        print(f"pearson r = {report['pearson_r']:.3f}, "
              f"within-1 = {report['within_one_point_frac']:.2%}")


if __name__ == "__main__":
    main()
