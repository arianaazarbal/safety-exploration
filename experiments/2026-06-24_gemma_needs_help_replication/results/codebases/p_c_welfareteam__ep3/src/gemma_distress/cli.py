"""Command-line entry point: ``gemma-distress <command> ...`` (or ``python -m
gemma_distress``).

Thin argparse dispatcher over the ``pipelines`` package. Each command maps to one
paper section; run them in order (elicit -> judge -> analyze -> ...). All commands
accept ``--config`` / ``--models-config`` to point at alternative YAML files, and
read API keys / the run directory from the environment (see ``.env.example``).

Typical full run::

    gemma-distress elicit --model gemma-3-27b-it
    gemma-distress judge  --model gemma-3-27b-it
    gemma-distress analyze
    gemma-distress agreement
    gemma-distress prefill all
    gemma-distress calm all
    gemma-distress build-datasets all
    gemma-distress train dpo
    # register runs/section4/dpo_model as a target with adapter_path, then:
    gemma-distress petri run
    gemma-distress capabilities run --model gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipelines import (
    agreement,
    analyze,
    calm,
    capabilities,
    datasets,
    elicit,
    petri,
    prefill,
    score,
    train,
)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="experiment YAML (default: configs/default.yaml)")
    p.add_argument("--models-config", default=None, help="model registry YAML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma-distress", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # Section 2
    p = sub.add_parser("elicit", help="run multi-turn rejection rollouts (Section 2)")
    _add_common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("judge", help="score rollouts on the 0-10 scale (Section 2.1)")
    _add_common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--judge-role", default="frustration_primary")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("agreement", help="inter-judge agreement validation (Section 2.1)")
    _add_common(p)
    p.add_argument("--models", nargs="*", default=None)

    p = sub.add_parser("analyze", help="aggregate tables / curves / words (Section 2.2)")
    _add_common(p)
    p.add_argument("--models", nargs="*", default=None)

    # Section 3
    p = sub.add_parser("prefill", help="base-vs-instruct prefilling (Section 3, Gemma)")
    _add_common(p)
    p.add_argument("action", choices=["prepare", "continue", "summarize", "all"])
    p.add_argument("--model", default=None, help="required for `continue`")

    # Section 4
    p = sub.add_parser("calm", help="generate+score calm/vanilla data (Section 4.1)")
    _add_common(p)
    p.add_argument("action", choices=["generate", "score", "all"])
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("build-datasets", help="build SFT/DPO datasets (Section 4.1)")
    _add_common(p)
    p.add_argument("action", choices=["sft", "dpo", "all"])

    p = sub.add_parser("train", help="train SFT/DPO LoRA adapters (Section 4.1)")
    _add_common(p)
    p.add_argument("action", choices=["dpo", "sft"])

    p = sub.add_parser("petri", help="open-ended emotion elicitation (Section 4.2)")
    _add_common(p)
    p.add_argument("action", choices=["run", "summarize", "all"])
    p.add_argument("--models", nargs="*", default=None)

    p = sub.add_parser("capabilities", help="capability-preservation benchmarks (Section 4.2)")
    _add_common(p)
    p.add_argument("action", choices=["run", "summarize", "all"])
    p.add_argument("--model", default=None, help="required for `run`")
    p.add_argument("--models", nargs="*", default=None, help="for `summarize`")
    p.add_argument("--benchmarks", nargs="*", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.models_config)
    cmd = args.command

    if cmd == "elicit":
        elicit.run(config, args.model, seed=args.seed, limit=args.limit)
    elif cmd == "judge":
        score.run(config, args.model, judge_role=args.judge_role, limit=args.limit)
    elif cmd == "agreement":
        agreement.run(config, models=args.models)
    elif cmd == "analyze":
        analyze.run(config, models=args.models)
    elif cmd == "prefill":
        if args.action == "prepare":
            prefill.prepare(config)
        elif args.action == "continue":
            if not args.model:
                raise SystemExit("prefill continue requires --model")
            prefill.continue_model(config, args.model)
        elif args.action == "summarize":
            prefill.summarize(config)
        else:
            prefill.run(config)
    elif cmd == "calm":
        if args.action == "generate":
            calm.generate(config, limit=args.limit)
        elif args.action == "score":
            calm.score(config)
        else:
            calm.run(config, limit=args.limit)
    elif cmd == "build-datasets":
        if args.action == "sft":
            datasets.build_sft(config)
        elif args.action == "dpo":
            datasets.build_dpo(config)
        else:
            datasets.run(config)
    elif cmd == "train":
        if args.action == "dpo":
            train.train_dpo_model(config)
        else:
            train.train_sft_model(config)
    elif cmd == "petri":
        if args.action == "run":
            for m in (args.models or config.all_targets()):
                petri.run_model(config, m)
        elif args.action == "summarize":
            petri.summarize(config, models=args.models)
        else:
            petri.run(config, models=args.models)
    elif cmd == "capabilities":
        if args.action == "run":
            if not args.model:
                raise SystemExit("capabilities run requires --model")
            capabilities.run_model(config, args.model, benchmarks=args.benchmarks)
        elif args.action == "summarize":
            capabilities.summarize(config, models=args.models)
        else:
            for m in (args.models or config.all_targets()):
                capabilities.run_model(config, m, benchmarks=args.benchmarks)
            capabilities.summarize(config, models=args.models)
    else:  # pragma: no cover
        raise SystemExit(f"unknown command {cmd}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
